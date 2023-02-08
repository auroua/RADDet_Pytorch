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
from model.yolo_head import decodeYolo, yoloheadToPredictions, nms, nmsOverClass
from dataset.radar_dataset_plot import RararDatasetEvaluate
from dataset.radar_dataset_eval_tf import RararDatasetEvaluateTF


device = "cuda" if torch.cuda.is_available() else "cpu"


def getTruePositive(pred, gt, input_size, class_names, mode, iou_threshold=0.5):
    """ output tp (true positive) with size [num_pred, ] """
    assert mode in ["3D", "2D"]

    if mode == "3D":
        df = pd.DataFrame(gt, columns=['Range', 'Azimuth', 'Doppler', 'Range_w', 'Azimuth_w', 'Doppler_w', 'Class_ID'])
        df_pred = pd.DataFrame(pred, columns=['Range', 'Azimuth', 'Doppler', 'Range_w', 'Azimuth_w', 'Doppler_w',
                                              'Score_pred', 'Class_ID_pred'])
    else:
        df = pd.DataFrame(gt, columns=['Range', 'Azimuth', 'Range_w', 'Azimuth_w', 'Class_ID'])
        df_pred = pd.DataFrame(pred,
                               columns=['Range', 'Azimuth', 'Range_w', 'Azimuth_w', 'Score_pred', 'Class_ID_pred'])

    # initialize gt dataframe
    df['Class'] = 'x'  # gt class, will be filled later
    df['Score_pred'] = 0
    df['Class_ID_pred'] = len(class_names) + 1  # prediction class ID, set to background if the target is missed
    df['Class_pred'] = 'background'  # prediction class name, set to background if the target is missed
    df['IoU'] = 0
    df['TF'] = 'MD'  # default missed, not been detected
    df['Class_ID_pred'] = df['Class_ID_pred'].astype('int')
    df['Class_ID'] = df['Class_ID'].astype('int')
    for i in range(len(gt)):
        df.loc[i, 'Class'] = class_names[df['Class_ID'][i]]

    # initialize pred dataframe
    df_pred['Class_pred'] = 'x'  # predicted class, will be filled later
    df_pred['Class_ID'] = len(class_names) + 1  # gt class ID, default background
    df_pred['Class_ID'] = df_pred['Class_ID'].astype('int')
    df_pred['Class'] = 'background'  # gt class. set to background if prediction missed gt
    df_pred['IoU'] = 0
    df_pred['TF'] = 'FA'  # default false alarm, does not hit any targets
    df_pred['Class_ID_pred'] = df_pred['Class_ID_pred'].astype('int')
    for i in range(len(pred)):
        df_pred.loc[i, 'Class_pred'] = class_names[df_pred['Class_ID_pred'][i]]  # prediction class.

    detected_gt_boxes = []
    for i in range(len(pred)):
        current_pred = pred[i]
        if mode == "3D":
            current_pred_box = current_pred[:6]
            current_pred_score = current_pred[6]
            current_pred_class = current_pred[7].astype('int')
            gt_box = gt[..., :6]
            gt_class = gt[..., 6].astype('int')

            iou = helper.iou3d(current_pred_box[np.newaxis, ...], gt_box, input_size)
        else:
            current_pred_box = current_pred[:4]
            current_pred_score = current_pred[4]
            current_pred_class = current_pred[5].astype('int')
            gt_box = gt[..., :4]
            gt_class = gt[..., 4].astype('int')

            iou = helper.iou2d(current_pred_box[np.newaxis, ...], gt_box)

        iou_max_idx = np.argmax(iou)
        iou_max = iou[iou_max_idx]
        if iou_max > 0.0:
            df.loc[iou_max_idx, 'Score_pred'] = current_pred_score
            df.loc[iou_max_idx, 'Class_ID_pred'] = current_pred_class
            df.loc[iou_max_idx, 'Class_pred'] = class_names[current_pred_class]
            df.loc[iou_max_idx, 'IoU'] = iou_max

            df_pred.loc[i, 'Class_ID'] = gt_class[iou_max_idx]
            df_pred.loc[i, 'Class'] = class_names[gt_class[iou_max_idx]]
            df_pred.loc[i, 'IoU'] = iou_max
            if (iou_max >= iou_threshold) and (current_pred_class == gt_class[iou_max_idx]) and (
                    iou_max_idx not in detected_gt_boxes):
                df.loc[iou_max_idx, 'TF'] = 'TP'  # true positive, class match and IoU > 0.5
                df_pred.loc[i, 'TF'] = 'TP'  # true positive
            else:
                df.loc[iou_max_idx, 'TF'] = 'FN'  # false negative
                df_pred.loc[i, 'TF'] = 'FP'  # false positive

            detected_gt_boxes.append(iou_max_idx)

    # df_FA = df_pred[df_pred['TF']=='FA']            # false alarm = false positive with IoU=0
    # if len(df_FA) > 0:
    #     df = pd.concat([df, df_FA], ignore_index=True)

    df_MD = df[df['TF'] == 'MD']  # missed gt objection
    if len(df_MD) > 0:
        df_pred = pd.concat([df_pred, df_MD], ignore_index=True)

    return df_pred


def get_GT_TF(predictions, gts, input_size, class_names, mode, tp_iou_threshold=0.5):
    """
    mode="3D":
        predictions         ->      [num_pred, 6 + score + class]
        gts                 ->      [num_gt, 6 + class]

    mode="2D":
        predictions         ->      [num_pred, 4 + score + class]
        gts                 ->      [num_gt, 4 + class]
    """
    assert mode in ["3D", "2D"]

    if mode == "3D":
        gts = gts[gts[..., :6].any(axis=-1) > 0]
        ### NOTE: sort prediction using scores ###
        sorted_idx = np.argsort(predictions[..., 6])[::-1]
    else:
        gts = gts[gts[..., :4].any(axis=-1) > 0]
        ### NOTE: sort prediction using scores ###
        sorted_idx = np.argsort(predictions[..., 4])[::-1]

    pred = predictions[sorted_idx]
    df = getTruePositive(pred, gts, input_size, class_names, mode, iou_threshold=tp_iou_threshold)
    return df


def main(args):
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

    if_evaluate_cart = True
    logdir = os.path.join(config_evaluate["log_dir"], config["NAME"] + "-" + config_data["RAD_dir"] +
                          "-b_" + str(config_train["batch_size"]) +
                          "-lr_" + str(config_train["learningrate_init"]))
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logdir_cart = os.path.join(config_train["log_dir"], config["NAME"] + "-" + config_data["RAD_dir"] +
                               "-b_" + str(config_train["batch_size"]) +
                               "-lr_" + str(config_train["learningrate_init"]) + "_cartesian")
    if not os.path.exists(logdir_cart):
        os.makedirs(logdir_cart)

    output_dir = os.path.join(config_evaluate["log_dir"], 'results_TrueFalse')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    radar_dataset_plot = RararDatasetEvaluateTF(
        config_data=config_data,
        config_train=config_train,
        config_model=config_model,
        config_radar=config_radar,
        headoutput_shape=config_data["headoutput_shape"],
        anchors=anchor_boxes_bk,
        anchors_cart=anchor_boxes_cart_bk,
        cart_shape=config_data["cart_shape"],
        dType="test"
    )

    input_size = torch.tensor(list(config_model["input_shape"]), dtype=torch.float32).to(device)
    anchor_boxes = torch.tensor(anchor_boxes, dtype=torch.float32).to(device)

    df_3D_TF = pd.DataFrame()
    df_2D_TF = pd.DataFrame()

    print("Start plotting, it might take a while...")
    for RAD_file, data, label, label_cart, raw_boxes, raw_boxes_cart in radar_dataset_plot:
        data = data.to(device)
        label = label.to(device)
        raw_boxes = raw_boxes.cpu().numpy()
        raw_boxes_cart = raw_boxes_cart.cpu().numpy()

        _, feature = model(data)
        pred_raw, pred = decodeYolo(feature,
                                    input_size=input_size,
                                    anchor_boxes=anchor_boxes,
                                    scale=config_model["yolohead_xyz_scales"][0])
        pred = pred.cpu().detach().numpy()
        pred_frame = pred[0]
        predicitons = yoloheadToPredictions(pred_frame,
                                            conf_threshold=config_model["confidence_threshold"])
        nms_pred = nmsOverClass(predicitons, config_model["nms_iou3d_threshold"],
                                config_model["input_shape"], sigma=0.3, method="nms")
        if if_evaluate_cart:
            pred_raw = model_cart(data)
            pred = model_cart.decodeYolo(pred_raw)
            # raw_boxes = raw_boxes.cpu().numpy()
            pred = pred.cpu().detach().numpy()
            pred_frame = pred[0]
            predicitons = yoloheadToPredictions2D(pred_frame,
                                                  conf_threshold=0.5)
            nms_pred_cart = helper.nms2DOverClass(predicitons,
                                                  config_evaluate["nms_iou3d_threshold"],
                                                  config_model["input_shape"], sigma=0.3, method="nms")
        else:
            nms_pred_cart = None

        dataID = os.path.splitext(os.path.basename(RAD_file))[0]

        df = get_GT_TF(nms_pred_cart, raw_boxes_cart, config_model["input_shape"], config_data["all_classes"],
                       mode="2D")
        df.insert(0, 'Image_ID', dataID)
        df_2D_TF = pd.concat([df_2D_TF, df], ignore_index=True)

        df = get_GT_TF(nms_pred, raw_boxes, config_model["input_shape"], config_data["all_classes"], mode="3D")
        df.insert(0, 'Image_ID', dataID)
        df_3D_TF = pd.concat([df_3D_TF, df], ignore_index=True)

    filename = os.path.join(output_dir, config_data["RAD_dir"] + "-RAD Boxes GT TF.csv")
    df_3D_TF.to_csv(filename, index=False, float_format="%.3f")

    filename = os.path.join(output_dir, config_data["RAD_dir"] + "-Cartesian Boxes GT TF.csv")
    df_2D_TF.to_csv(filename, index=False, float_format="%.3f")


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
    main(args)

