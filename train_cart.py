import torch.optim

import util.loader as loader
import argparse
import os
import sys
import numpy as np
from engine.launch import launch
from model.model import RADDet
from utils.collect_env import collect_env_info
from utils.dist_utils import get_rank
from dataset.radar_dataset_3d import RararDataset3D
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils.optimizer_utils import LinearWarmupCosineAnnealingLR
from metrics import mAP
import utils.dist_utils as dist_utils
from model.model_cart import RADDetCart
from model.yolo_loss import yoloheadToPredictions2D, nms2DOverClass


def main(args):
    # initialization
    # np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    env_str = collect_env_info()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(env_str)

    config = loader.readConfig(config_file_name="./config.json")
    config_data = config["DATA"]
    config_radar = config["RADAR_CONFIGURATION"]
    config_model = config["MODEL"]
    config_train = config["TRAIN"]

    # load anchor boxes with order
    anchor_boxes = loader.readAnchorBoxes(anchor_boxes_file="./anchors.txt")
    num_classes = len(config_data["all_classes"])

    anchor_boxes_cart = loader.readAnchorBoxes(anchor_boxes_file="./anchors_cartboxes.txt")


    ### NOTE: using the yolo head shape out from model for data generator ###
    model = RADDet(config_model, config_data, config_train, anchor_boxes)
    # model.to(device)
    print(f"Load pretrained model from {args.backbone_resume_from}")
    model.load_state_dict(torch.load(args.backbone_resume_from))

    model_bk = model.backbone

    model_cart = RADDetCart(config_model, config_data, config_train, anchor_boxes_cart, config_model["bk_output_size"],
                            device, backbone=model_bk)
    model_cart.to(device)

    train_dataset = RararDataset3D(config_data=config_data,
                                   config_train=config_train,
                                   config_model=config_model,
                                   headoutput_shape=config_data["headoutput_shape"],
                                   anchors=anchor_boxes,
                                   anchors_cart=anchor_boxes_cart,
                                   cart_shape=config_data["cart_shape"],
                                   dType="train")

    validate_dataset = RararDataset3D(config_data=config_data,
                                      config_train=config_train,
                                      config_model=config_model,
                                      headoutput_shape=config_data["headoutput_shape"],
                                      anchors=anchor_boxes,
                                      anchors_cart=anchor_boxes_cart,
                                      cart_shape=config_data["cart_shape"],
                                      dType="validate")

    train_loader = DataLoader(train_dataset,
                              batch_size=config_train["batch_size"]//args.num_gpus,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              persistent_workers=True)

    validate_loader = DataLoader(validate_dataset,
                                 batch_size=config_train["batch_size"]//args.num_gpus,
                                 shuffle=True,
                                 num_workers=4,
                                 pin_memory=True,
                                 persistent_workers=True)

    test_dataset = RararDataset3D(config_data=config_data,
                                  config_train=config_train,
                                  config_model=config_model,
                                  headoutput_shape=config_data["headoutput_shape"],
                                  anchors=anchor_boxes,
                                  anchors_cart=anchor_boxes_cart,
                                  cart_shape=config_data["cart_shape"],
                                  dType="test")

    test_loader = DataLoader(test_dataset,
                             batch_size=config_train["batch_size"]//args.num_gpus,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True,
                             persistent_workers=True)
    if get_rank() == 0:
        ### NOTE: training settings ###
        # logdir = os.path.join(config_train["log_dir"], "cartesian",
        #                       "b_" + str(config_train["batch_size"]) + "lr_" + str(config_train["learningrate_init"]))
        logdir = os.path.join(config_train["log_dir"], config["NAME"] + "-" + config_data["RAD_dir"] +
                              "-b_" + str(config_train["batch_size"]) +
                              "-lr_" + str(config_train["learningrate_init"]) + "_cartesian")
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        writer = SummaryWriter(logdir=logdir)
        log_specific_dir = os.path.join(logdir, "ckpt")
        if not os.path.exists(log_specific_dir):
            os.makedirs(log_specific_dir)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config_train["learningrate_init"], betas=(0.9, 0.99))

    scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                              warmup_epochs=config_train["warmup_steps"],
                                              eta_min=config_train["learningrate_end"],
                                              max_epochs=config_train["epochs"],
                                              warmup_start_lr=config_train["learningrate_init"])
    step, best_val = 0, 0.0
    for i in range(config_train["epochs"]):
        for d in train_loader:
            data, label, raw_boxes = d
            data = data.to(device)
            label = label.to(device)
            raw_boxes = raw_boxes.to(device)
            # print(data.shape, label.shape, raw_boxes.shape, data.device)
            pred_raw = model_cart(data)
            pred = model_cart.decodeYolo(pred_raw)
            total_loss, box_loss, conf_loss, category_loss = model_cart.loss(pred_raw, pred, label, raw_boxes[..., :4])
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_r, box_loss_r, conf_loss_r, category_loss_r = total_loss.cpu().detach(), \
                box_loss.cpu().detach(), conf_loss.cpu().detach(), category_loss.cpu().detach()

            if get_rank() == 0:
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=step)
                writer.add_scalar("loss/total_loss", total_loss_r, global_step=step)
                writer.add_scalar("loss/box_loss", box_loss_r, global_step=step)
                writer.add_scalar("loss/conf_loss", conf_loss_r, global_step=step)
                writer.add_scalar("loss/category_loss", category_loss_r, global_step=step)
                writer.flush()
            step += 1

            if get_rank() == 0:
                print("=======> epochs: %4d, train step: %4d, lr: %.10f, total_loss: %4.2f, "
                      "box_loss: %4.2f, conf_loss: %4.2f, category_loss: %4.2f" %
                      (i, step, optimizer.param_groups[0]['lr'], total_loss_r, box_loss_r, conf_loss_r,
                       category_loss_r))
        if get_rank() == 0:
            print(f"epochs: {i}, start validation")
            mean_ap_test = 0.0
            ap_all_class_test = []
            ap_all_class = []
            total_losstest = []
            box_losstest = []
            conf_losstest = []
            category_losstest = []
            for class_id in range(num_classes):
                ap_all_class.append([])
            model.eval()
            for d in validate_loader:
                with torch.no_grad():
                    data, label, raw_boxes = d
                    data = data.to(device)
                    label = label.to(device)
                    raw_boxes = raw_boxes.to(device)
                    # feature = model_bk(data)
                    pred_raw = model_cart(data)
                    pred = model_cart.decodeYolo(pred_raw)
                    total_loss, box_loss, conf_loss, category_loss = model_cart.loss(pred_raw, pred, label,
                                                                                     raw_boxes[..., :4])
                    box_loss_b, conf_loss_b, category_loss_b = box_loss.cpu().detach(), conf_loss.cpu().detach(), \
                        category_loss.cpu().detach()
                    total_loss_b = total_loss.cpu().detach()
                    total_losstest.append(total_loss_b)
                    box_losstest.append(box_loss_b)
                    conf_losstest.append(conf_loss_b)
                    category_losstest.append(category_loss_b)
                    raw_boxes = raw_boxes.cpu().numpy()
                    pred = pred.cpu().detach().numpy()
                    for batch_id in range(raw_boxes.shape[0]):
                        raw_boxes_frame = raw_boxes[batch_id]
                        pred_frame = pred[batch_id]
                        predicitons = yoloheadToPredictions2D(pred_frame,
                                                            conf_threshold=config_model["confidence_threshold"])
                        nms_pred = nms2DOverClass(predicitons, config_model["nms_iou3d_threshold"],
                                       config_model["input_shape"], sigma=0.3, method="nms")
                        mean_ap, ap_all_class = mAP.mAP2D(nms_pred, raw_boxes_frame,
                                                        config_model["input_shape"], ap_all_class,
                                                        tp_iou_threshold=config_model["mAP_iou3d_threshold"])
                        mean_ap_test += mean_ap
            for ap_class_i in ap_all_class:
                if len(ap_class_i) == 0:
                    class_ap = 0.
                else:
                    class_ap = np.mean(ap_class_i)
                ap_all_class_test.append(class_ap)
            mean_ap_test /= len(validate_loader)
            print("-------> ap: %.6f" % mean_ap_test)

            writer.add_scalar("ap/ap_all", mean_ap_test, global_step=i)
            writer.add_scalar("ap/ap_person", ap_all_class_test[0], global_step=i)
            writer.add_scalar("ap/ap_bicycle", ap_all_class_test[1], global_step=i)
            writer.add_scalar("ap/ap_car", ap_all_class_test[2], global_step=i)
            writer.add_scalar("ap/ap_motorcycle", ap_all_class_test[3], global_step=i)
            writer.add_scalar("ap/ap_bus", ap_all_class_test[4], global_step=i)
            writer.add_scalar("ap/ap_truck", ap_all_class_test[5], global_step=i)
            ### NOTE: validate loss ###
            writer.add_scalar("validate_loss/total_loss",
                              np.mean(total_losstest), global_step=i)
            writer.add_scalar("validate_loss/box_loss",
                              np.mean(box_losstest), global_step=i)
            writer.add_scalar("validate_loss/conf_loss",
                              np.mean(conf_losstest), global_step=i)
            writer.add_scalar("validate_loss/category_loss",
                              np.mean(category_losstest), global_step=i)
            writer.flush()

            ckpt_file_path = os.path.join(log_specific_dir, f"{i}.pth")
            ckpt_file_best_path = os.path.join(log_specific_dir, "best.pth")
            with open(ckpt_file_path, 'wb') as f:
                torch.save(model_cart.state_dict(), f)
            print("Saving checkpoint to {}".format(ckpt_file_path))
            if mean_ap_test > best_val:
                best_val = mean_ap_test
                with open(ckpt_file_best_path, 'wb') as f:
                    torch.save(model_cart.state_dict(), f)
            print("Current model is the best model at present, and it is saved to {}".format(ckpt_file_best_path))
        if dist_utils.get_world_size() > 1 and dist_utils.get_world_size() > 0:
            torch.distributed.barrier()
        model.train()
        scheduler.step()


def get_parse():
    parser = argparse.ArgumentParser(description='Args for segmentation model.')
    parser.add_argument("--num-gpus", type=int,
                        default=1,
                        help="The number of gpus.")
    parser.add_argument("--num-machines", type=int,
                        default=1,
                        help="The number of machines.")
    parser.add_argument("--machine-rank", type=int,
                        default=0,
                        help="The rank of current machine.")
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument("--dist_url", type=str,
                        default="tcp://127.0.0.1:{}".format(port),
                        help="initialization URL for pytorch distributed backend.")
    parser.add_argument("--backbone_resume_from", type=str,
                        default="/home/albert_wei/WorkSpaces_2023/RADIA/RADDet/RADDet_Pytorch/logs/RadarResNet/b_4lr_0.0001/ckpt/best.pth",
                        help="The number of machines.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parse()
    print("Command Line Args: ", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )