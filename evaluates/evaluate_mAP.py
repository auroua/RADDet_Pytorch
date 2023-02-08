# Title: RADDet
# Authors: Ao Zhang, Erlik Nowruzi, Robert Laganiere
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import shutil
from tabulate import tabulate
import util.helper as helper
import util.drawer as drawer

import pandas as pd

import torch.optim
from dataset.radar_dataset import RararDataset
import util.loader as loader
import argparse
import os
import sys
import numpy as np
from model.model import RADDet
from dataset.radar_dataset_3d import RararDataset3D
from torch.utils.data import DataLoader
from metrics import mAP
from model.model_cart import RADDetCart
from model.yolo_loss import yoloheadToPredictions2D
import copy
from model.yolo_head import decodeYolo, yoloheadToPredictions, nms
from dataset.radar_dataset_plot import RararDatasetEvaluate


device = "cuda" if torch.cuda.is_available() else "cpu"

def cutImage(image_dir, image_filename):
    image_name = os.path.join(image_dir, image_filename)
    image = cv2.imread(image_name)
    part_1 = image[:, 1540:1750, :]
    part_2 = image[:, 2970:3550, :]
    part_3 = image[:, 4370:5400, :]
    part_4 = image[:, 6200:6850, :]
    new_img = np.concatenate([part_4, part_1, part_2, part_3], axis=1)
    cv2.imwrite(image_name, new_img)


def cutImage3Axes(image_dir, image_filename):
    image_name = os.path.join(image_dir, image_filename)
    image = cv2.imread(image_name)
    part_1 = image[:, 1780:2000, :]
    part_2 = image[:, 3800:4350, :]
    part_3 = image[:, 5950:6620, :]
    new_img = np.concatenate([part_3, part_1, part_2], axis=1)
    cv2.imwrite(image_name, new_img)


### NOTE: define testing step for RAD Boxes Model ###
def test_step(config_model, model, test_dataloader, num_classes, input_size,
              anchor_boxes, map_iou_threshold_list):
    mean_ap_test_all = []
    ap_all_class_test_all = []
    ap_all_class_all = []
    for i in range(len(map_iou_threshold_list)):
        mean_ap_test_all.append(0.0)
        ap_all_class_test_all.append([])
        ap_all_class = []
        for class_id in range(num_classes):
            ap_all_class.append([])
        ap_all_class_all.append(ap_all_class)
    print("Start evaluating RAD Boxes on the entire dataset, it might take a while...")
    # pbar = tqdm(total=int(data_generator.total_test_batches))
    for data, label, raw_boxes in test_dataloader:
        data = data.to(device)
        label = label.to(device)
        raw_boxes = raw_boxes.to(device)

        _, feature = model(data)
        pred_raw, pred = decodeYolo(feature,
                                    input_size=input_size,
                                    anchor_boxes=anchor_boxes,
                                    scale=config_model["yolohead_xyz_scales"][0])
        pred = pred.cpu().detach().numpy()
        raw_boxes = raw_boxes.cpu().numpy()
        for batch_id in range(raw_boxes.shape[0]):
            raw_boxes_frame = raw_boxes[batch_id]
            pred_frame = pred[batch_id]
            predicitons = yoloheadToPredictions(pred_frame,
                                                conf_threshold=config_model["confidence_threshold"])
            nms_pred = nms(predicitons, config_model["nms_iou3d_threshold"],
                           config_model["input_shape"], sigma=0.3, method="nms")
            for j in range(len(map_iou_threshold_list)):
                map_iou_threshold = map_iou_threshold_list[j]
                mean_ap, ap_all_class_all[j] = mAP.mAP(nms_pred, raw_boxes_frame,
                                                       config_model["input_shape"],
                                                       ap_all_class_all[j],
                                                       tp_iou_threshold=map_iou_threshold)
                mean_ap_test_all[j] += mean_ap

    for iou_threshold_i in range(len(map_iou_threshold_list)):
        ap_all_class = ap_all_class_all[iou_threshold_i]
        for ap_class_i in ap_all_class:
            if len(ap_class_i) == 0:
                class_ap = 0.
            else:
                class_ap = np.mean(ap_class_i)
            ap_all_class_test_all[iou_threshold_i].append(class_ap)
        mean_ap_test_all[iou_threshold_i] /= len(test_dataloader.dataset)
    return mean_ap_test_all, ap_all_class_test_all


## NOTE: define testing step for Cartesian Boxes Model ###
def test_step_cart(config_model, config_evaluate, model_cart, test_dataloader, num_classes, map_iou_threshold_list):
    mean_ap_test_all = []
    ap_all_class_test_all = []
    ap_all_class_all = []
    for i in range(len(map_iou_threshold_list)):
        mean_ap_test_all.append(0.0)
        ap_all_class_test_all.append([])
        ap_all_class = []
        for class_id in range(num_classes):
            ap_all_class.append([])
        ap_all_class_all.append(ap_all_class)
    print("Start evaluating Cartesian Boxes on the entire dataset, it might take a while...")
    for data, label, raw_boxes in test_dataloader:
        data = data.to(device)
        label = label.to(device)
        raw_boxes = raw_boxes.to(device)

        pred_raw = model_cart(data)
        pred = model_cart.decodeYolo(pred_raw)

        raw_boxes = raw_boxes.cpu().numpy()
        pred = pred.cpu().detach().numpy()
        for batch_id in range(raw_boxes.shape[0]):
            raw_boxes_frame = raw_boxes[batch_id]
            pred_frame = pred[batch_id]
            predicitons = yoloheadToPredictions2D(pred_frame,
                                                  conf_threshold=0.5)
            nms_pred = helper.nms2D(predicitons,
                                    config_evaluate["nms_iou3d_threshold"],
                                    config_model["input_shape"], sigma=0.3, method="nms")
            for j in range(len(map_iou_threshold_list)):
                map_iou_threshold = map_iou_threshold_list[j]
                mean_ap, ap_all_class_all[j] = mAP.mAP2D(nms_pred, raw_boxes_frame, config_model["input_shape"],
                                                         ap_all_class_all[j],
                                                         tp_iou_threshold=map_iou_threshold)
                mean_ap_test_all[j] += mean_ap
    for iou_threshold_i in range(len(map_iou_threshold_list)):
        ap_all_class = ap_all_class_all[iou_threshold_i]
        for ap_class_i in ap_all_class:
            if len(ap_class_i) == 0:
                class_ap = 0.
            else:
                class_ap = np.mean(ap_class_i)
            ap_all_class_test_all[iou_threshold_i].append(class_ap)
        mean_ap_test_all[iou_threshold_i] /= len(test_dataloader.dataset)
    return mean_ap_test_all, ap_all_class_test_all

def predictionPlots(config_data, if_evaluate_cart, model, model_cart, radar_dataset,
                    config_evaluate, config_model, input_size, anchor_boxes):
    """ Plot the predictions of all data in dataset """
    if if_evaluate_cart:
        fig, axes = drawer.prepareFigure(4, figsize=(80, 6))
    else:
        fig, axes = drawer.prepareFigure(3, figsize=(80, 6))
    # colors = loader.randomColors(config_data["all_classes"])
    gt_colors = drawer.SetColorsGreen(len(config_data["all_classes"]))
    pred_colors = drawer.SetColorsRed(len(config_data["all_classes"]))

    image_save_dir = "./images/evaluate_plots/" + config_data["RAD_dir"]
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
    else:
        shutil.rmtree(image_save_dir)
        os.makedirs(image_save_dir)
    print("Start plotting, it might take a while...")
    for RAD_file, data, label, label_cart, raw_boxes, raw_boxes_cart, \
            stereo_left_image, RD_img, RA_img, RA_cart_img, gt_instances in radar_dataset:
        data = data.to(device)
        feature_bk, feature = model(data)
        pred_raw, pred = decodeYolo(feature,
                                    input_size=input_size,
                                    anchor_boxes=anchor_boxes,
                                    scale=config_model["yolohead_xyz_scales"][0])
        pred = pred.cpu().detach().numpy()
        pred_frame = pred[0]
        predicitons = helper.yoloheadToPredictions(pred_frame, conf_threshold=0.5)
        nms_pred = helper.nmsOverClass(predicitons, config_evaluate["nms_iou3d_threshold"],
                                       config_model["input_shape"], sigma=0.3, method="nms")

        if if_evaluate_cart:
            pred_raw_cart = model_cart(data)
            pred_cart = model_cart.decodeYolo(pred_raw_cart)
            pred_cart = pred_cart.cpu().detach().numpy()
            pred_frame_cart = pred_cart[0]
            predicitons_cart = helper.yoloheadToPredictions2D(pred_frame_cart,
                                                              conf_threshold=0.5)
            nms_pred_cart = helper.nms2DOverClass(predicitons_cart, \
                                                  config_model["nms_iou3d_threshold"], \
                                                  config_model["input_shape"], \
                                                  sigma=0.3, method="nms")
        else:
            nms_pred_cart = None

        drawer.clearAxes(axes)
        drawer.drawRadarPredWithGt(stereo_left_image, RD_img, \
                                   RA_img, RA_cart_img, gt_instances, nms_pred, \
                                   config_data["all_classes"], gt_colors, pred_colors, axes, \
                                   radar_cart_nms=nms_pred_cart)
        dataID = os.path.splitext(os.path.basename(RAD_file))[0]
        drawer.saveFigure(image_save_dir, "%s.png" % (dataID))
        if if_evaluate_cart:
            cutImage(image_save_dir, "%s.png" % (dataID))
        else:
            cutImage3Axes(image_save_dir, "%s.png" % (dataID))


def main(args, RAD_dir='RAD'):
    # initialization
    # np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    # env_str = collect_env_info()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(env_str)

    config = loader.readConfig(config_file_name="../config.json")
    config_data = config["DATA"]
    config_radar = config["RADAR_CONFIGURATION"]
    config_model = config["MODEL"]
    config_train = config["TRAIN"]
    config_evaluate = config["EVALUATE"]

    config_data['RAD_dir'] = RAD_dir

    # load anchor boxes with order
    anchor_boxes = loader.readAnchorBoxes(anchor_boxes_file="../anchors.txt")
    num_classes = len(config_data["all_classes"])
    anchor_boxes_cart = loader.readAnchorBoxes(anchor_boxes_file="../anchors_cartboxes.txt")

    anchor_boxes_bk = copy.deepcopy(anchor_boxes)
    anchor_boxes_cart_bk = copy.deepcopy(anchor_boxes_cart)

    ### NOTE: using the yolo head shape out from model for data generator ###
    model = RADDet(config_model, config_data, config_train, anchor_boxes)
    model.load_state_dict(torch.load(args.resume_from))
    model_bk = copy.deepcopy(model.backbone)
    model.to(device)

    model_cart = RADDetCart(config_model, config_data, config_train, anchor_boxes_cart, config_model["bk_output_size"],
                            device, backbone=model_bk)
    model_cart.load_state_dict(torch.load(args.cart_resume_from))
    model_cart.to(device)

    test_dataset = RararDataset(config_data, config_train, config_model,
                                config_model["feature_out_shape"], anchor_boxes, dType="test", RADDir=RAD_dir)    # 2032
    test_loader = DataLoader(test_dataset,
                             batch_size=config_train["batch_size"]//args.num_gpus,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True,
                             persistent_workers=True)

    test_dataset_cart = RararDataset3D(config_data=config_data,
                                       config_train=config_train,
                                       config_model=config_model,
                                       headoutput_shape=config_data["headoutput_shape"],
                                       anchors=anchor_boxes,
                                       anchors_cart=anchor_boxes_cart,
                                       cart_shape=config_data["cart_shape"],
                                       dType="test",
                                       RADDir=RAD_dir)

    test_loader_cart = DataLoader(test_dataset_cart,
                                  batch_size=config_train["batch_size"]//args.num_gpus,
                                  shuffle=False,
                                  num_workers=4,
                                  pin_memory=True,
                                  persistent_workers=True)


    ### NOTE: RAD Boxes ckpt ###
    logdir = os.path.join(config_evaluate["log_dir"], config["NAME"] + "-" + config_data["RAD_dir"] +
                          "-b_" + str(config_train["batch_size"]) +
                          "-lr_" + str(config_train["learningrate_init"]))
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    ### NOTE: Cartesian Boxes ckpt ###
    if_evaluate_cart = True
    logdir_cart = os.path.join(config_train["log_dir"], config["NAME"] + "-" + config_data["RAD_dir"] +
                               "-b_" + str(config_train["batch_size"]) +
                               "-lr_" + str(config_train["learningrate_init"]) + "_cartesian")
    if not os.path.exists(logdir_cart):
        os.makedirs(logdir_cart)

    output_dir = os.path.join(config_evaluate["log_dir"], 'results_mAP')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    input_size = torch.tensor(list(config_model["input_shape"]), dtype=torch.float32).to(device)
    anchor_boxes = torch.tensor(anchor_boxes, dtype=torch.float32).to(device)

    ### NOTE: evaluate RAD Boxes under different mAP_iou ###
    all_mean_aps, all_ap_classes = test_step(config_model=config_model,
                                             model=model,
                                             test_dataloader=test_loader,
                                             num_classes=num_classes,
                                             input_size=input_size,
                                             anchor_boxes=anchor_boxes,
                                             map_iou_threshold_list=config_evaluate["mAP_iou3d_threshold"]
                                             )

    all_mean_aps = np.array(all_mean_aps)
    all_ap_classes = np.array(all_ap_classes)

    table = []
    row = []
    for i in range(len(all_mean_aps)):
        if i == 0:
            row.append("mAP")
        row.append(all_mean_aps[i])
    table.append(row)
    row = []
    for j in range(all_ap_classes.shape[1]):
        ap_current_class = all_ap_classes[:, j]
        for k in range(len(ap_current_class)):
            if k == 0:
                row.append(config_data["all_classes"][j])
            row.append(ap_current_class[k])
        table.append(row)
        row = []
    headers = []
    for ap_iou_i in config_evaluate["mAP_iou3d_threshold"]:
        if ap_iou_i == 0:
            headers.append("AP name")
        headers.append("AP_%.2f"%(ap_iou_i))
    print("==================== RAD Boxes AP ========================")
    print(tabulate(table, headers=headers))
    print("==========================================================")
    filename = os.path.join(output_dir, config_data["RAD_dir"]+"-RAD Boxes AP.csv")
    df = pd.DataFrame(table, columns=['metric']+[ap for ap in headers])
    df.to_csv(filename, index=False, float_format="%.3f")

    ### NOTE: evaluate Cart Boxes under different mAP_iou ###
    if if_evaluate_cart:
        all_mean_aps, all_ap_classes = test_step_cart(config_model=config_model,
                                                      config_evaluate=config_evaluate,
                                                      model_cart=model_cart,
                                                      test_dataloader=test_loader_cart,
                                                      num_classes=num_classes,
                                                      map_iou_threshold_list=config_evaluate["mAP_iou3d_threshold"]
                                                      )
        all_mean_aps = np.array(all_mean_aps)
        all_ap_classes = np.array(all_ap_classes)

        table = []
        row = []
        for i in range(len(all_mean_aps)):
            if i == 0:
                row.append("mAP")
            row.append(all_mean_aps[i])
        table.append(row)
        row = []
        for j in range(all_ap_classes.shape[1]):
            ap_current_class = all_ap_classes[:, j]
            for k in range(len(ap_current_class)):
                if k == 0:
                    row.append(config_data["all_classes"][j])
                row.append(ap_current_class[k])
            table.append(row)
            row = []
        headers = []
        for ap_iou_i in config_evaluate["mAP_iou3d_threshold"]:
            if ap_iou_i == 0:
                headers.append("AP name")
            headers.append("AP_%.2f" % (ap_iou_i))
        print("================= Cartesian Boxes AP =====================")
        print(tabulate(table, headers=headers))
        print("==========================================================")
        filename = os.path.join(output_dir, config_data["RAD_dir"] + "-Cartesian Boxes AP.csv")
        df = pd.DataFrame(table, columns=['metric'] + [ap for ap in headers])
        df.to_csv(filename, index=False, float_format="%.3f")

    radar_dataset_plot = RararDatasetEvaluate(
        config_data=config_data,
        config_train=config_train,
        config_model=config_model,
        config_radar=config_radar,
        headoutput_shape=config_data["headoutput_shape"],
        anchors=anchor_boxes_bk,
        anchors_cart=anchor_boxes_cart_bk,
        cart_shape=config_data["cart_shape"],
        dType="test",
        RADDir=RAD_dir
    )

    # radar_dataset_plot_loader = DataLoader(radar_dataset_plot,
    #                                        batch_size=1,
    #                                        shuffle=True,
    #                                        num_workers=4,
    #                                        pin_memory=True,
    #                                        persistent_workers=True)

    # NOTE: plot the predictions on the entire dataset ###
    predictionPlots(config_data, True, model, model_cart,
                    radar_dataset=iter(radar_dataset_plot),
                    config_evaluate=config_evaluate,
                    config_model=config_model,
                    input_size=input_size,
                    anchor_boxes=anchor_boxes,
                    )


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
    parser.add_argument("--resume_from", type=str,
                        default="/home/albert_wei/WorkSpaces_2023/RADIA/RADDet/RADDet_Pytorch/logs/RadarResNet/b_4lr_0.0001/ckpt/best.pth",
                        help="The number of machines.")
    parser.add_argument("--cart_resume_from", type=str,
                        default="/home/albert_wei/WorkSpaces_2023/RADIA/RADDet/RADDet_Pytorch/logs/RadarResNet/cartesian/b_4lr_0.0001/ckpt/best.pth",
                        help="The number of machines.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parse()
    # RAD_dir = 'RAD', 'RAD4', 'RAD_Sim8'
    main(args, RAD_dir='RAD')
    # main(args, RAD_dir='RAD4')
    # main(args, RAD_dir='RAD_Sim8')

