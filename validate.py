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
from dataset.radar_dataset import RararDataset
from torch.utils.data import DataLoader
from model.yolo_head import decodeYolo, yoloheadToPredictions, nms
from model.yolo_loss import RadDetLoss
from metrics import mAP


def main(args):
    # initialization
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    env_str = collect_env_info()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(env_str)

    config = loader.readConfig(config_file_name="./config.json")
    config_data = config["DATA"]
    config_model = config["MODEL"]
    config_train = config["TRAIN"]

    # load anchor boxes with order
    anchor_boxes = loader.readAnchorBoxes(anchor_boxes_file="./anchors.txt")
    num_classes = len(config_data["all_classes"])

    ### NOTE: using the yolo head shape out from model for data generator ###
    model = RADDet(config_model, config_data, config_train, anchor_boxes)
    # with open(args.resume_from, "rb") as f:
    print(f"Load pretrained model from {args.resume_from}")
    model.load_state_dict(torch.load(args.resume_from))
    model.to(device)

    model.eval()

    test_dataset = RararDataset(config_data, config_train, config_model,
                                config_model["feature_out_shape"], anchor_boxes, dType="test")    # 2032
    test_loader = DataLoader(test_dataset,
                             batch_size=config_train["batch_size"]//args.num_gpus,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True,
                             persistent_workers=True)
    if get_rank() == 0:
        ### NOTE: training settings ###
        logdir = os.path.join(config_train["log_dir"],
                              "b_" + str(config_train["batch_size"]) + "lr_" + str(config_train["learningrate_init"]))
        if not os.path.exists(logdir):
            os.makedirs(logdir)

    anchor_boxes = torch.tensor(anchor_boxes, dtype=torch.float32).to(device)
    input_size = torch.tensor(list(config_model["input_shape"]), dtype=torch.float32).to(device)

    criterion = RadDetLoss(
        input_size=input_size,
        focal_loss_iou_threshold=config_train["focal_loss_iou_threshold"]
    )

    print(f"start validation")
    mean_ap_test = 0.0
    ap_all_class_test = []
    ap_all_class = []
    total_losstest = []
    box_losstest = []
    conf_losstest = []
    category_losstest = []
    for class_id in range(num_classes):
        ap_all_class.append([])
    for d in test_loader:
        with torch.no_grad():
            data, label, raw_boxes = d
            data = data.to(device)
            label = label.to(device)
            raw_boxes = raw_boxes.to(device)
            _, feature = model(data)
            pred_raw, pred = decodeYolo(feature,
                                        input_size=input_size,
                                        anchor_boxes=anchor_boxes,
                                        scale=config_model["yolohead_xyz_scales"][0])
            box_loss, conf_loss, category_loss = criterion(pred_raw, pred, label, raw_boxes[..., :6])
            box_loss_b, conf_loss_b, category_loss_b = box_loss.cpu().detach(), conf_loss.cpu().detach(), \
                category_loss.cpu().detach()
            total_losstest.append(box_loss_b+conf_loss_b+category_loss_b)
            box_losstest.append(box_loss_b)
            conf_losstest.append(conf_loss_b)
            category_losstest.append(category_loss_b)
            raw_boxes = raw_boxes.cpu().numpy()
            pred = pred.cpu().detach().numpy()
            for batch_id in range(raw_boxes.shape[0]):
                raw_boxes_frame = raw_boxes[batch_id]
                pred_frame = pred[batch_id]
                predicitons = yoloheadToPredictions(pred_frame,
                                                    conf_threshold=config_model["confidence_threshold"])
                nms_pred = nms(predicitons, config_model["nms_iou3d_threshold"],
                               config_model["input_shape"], sigma=0.3, method="nms")
                mean_ap, ap_all_class = mAP.mAP(nms_pred, raw_boxes_frame,
                                                config_model["input_shape"], ap_all_class,
                                                tp_iou_threshold=config_model["mAP_iou3d_threshold"])
                mean_ap_test += mean_ap
    for ap_class_i in ap_all_class:
        if len(ap_class_i) == 0:
            class_ap = 0.
        else:
            class_ap = np.mean(ap_class_i)
        ap_all_class_test.append(class_ap)
    mean_ap_test /= len(test_dataset)
    print("-------> ap_all: %.6f, ap_person: %.6f, ap_bicycle: %.6f, ap_car: %.6f, ap_motorcycle: %.6f, ap_bus: %.6f, "
          "ap_truck: %.6f" % (mean_ap_test, ap_all_class_test[0], ap_all_class_test[1], ap_all_class_test[2],
                              ap_all_class_test[3], ap_all_class_test[4], ap_all_class_test[5]))
    print("-------> total_loss: %.6f, box_loss: %.6f, conf_loss: %.6f, category_loss: %.6f" %
          (np.mean(total_losstest), np.mean(box_losstest), np.mean(conf_losstest), np.mean(category_losstest)))


def get_parse():
    parser = argparse.ArgumentParser(description='Args for segmentation model.')
    parser.add_argument("--num-gpus", type=int,
                        default=1,
                        help="Inference code only support single GPU.")
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
    parser.add_argument("--resume_from", type=str,
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