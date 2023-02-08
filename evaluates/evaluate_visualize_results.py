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


    ### NOTE: RAD Boxes ckpt ###
    logdir = os.path.join(config_evaluate["log_dir"], config["NAME"] + "-" + config_data["RAD_dir"] +
                          "-b_" + str(config_train["batch_size"]) +
                          "-lr_" + str(config_train["learningrate_init"]))
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    ### NOTE: Cartesian Boxes ckpt ###
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

