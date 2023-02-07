from torch.utils.data import Dataset
import numpy as np
import os, glob
import util.loader as loader
import util.helper as helper
from torchvision.transforms import ToTensor
import torch


class RararDatasetEvaluate(Dataset):
    def __init__(self, config_data, config_train, config_model, config_radar, headoutput_shape,
                 anchors, transformer=ToTensor(), anchors_cart=None, cart_shape=None, dType="train"):
        super(RararDatasetEvaluate, self).__init__()
        self.input_size = config_model["input_shape"]
        self.config_data = config_data
        self.config_train = config_train
        self.config_model = config_model
        self.config_radar = config_radar
        self.headoutput_shape = headoutput_shape
        self.cart_shape = cart_shape
        self.grid_strides = self.getGridStrides()
        self.cart_grid_strides = self.getCartGridStrides()
        self.anchor_boxes = anchors
        self.anchor_boxes_cart = anchors_cart
        self.RAD_sequences_test = self.readSequences(mode="test")
        ### NOTE: if "if_validat" set true in "config.json", it will split trainset ###
        self.batch_size = config_train["batch_size"]
        self.total_test_batches = len(self.RAD_sequences_test) // self.batch_size
        self.dtype = dType
        self.transform = transformer

    def __len__(self):
        if self.dtype == "test":
            return len(self.RAD_sequences_test)
        else:
            raise ValueError("This type of dataset does not exist.")

    def __getitem__(self, index):
        if self.dtype == "test":
            return self.testData(index)
        else:
            raise ValueError("This type of dataset does not exist.")

    def testData(self, index):
        """ Generate test data with batch size """
        has_label = False
        while not has_label:
            RAD_filename = self.RAD_sequences_test[index]
            RAD_complex = loader.readRAD(RAD_filename)
            if RAD_complex is None:
                raise ValueError("RAD file not found, please double check the path")

            interpolation = 15
            RA = helper.getLog(helper.getSumDim(helper.getMagnitude(RAD_complex,
                                                                    power_order=2), target_axis=-1),
                               scalar=10, log_10=True)
            RD = helper.getLog(helper.getSumDim(helper.getMagnitude(RAD_complex,
                                                                    power_order=2), target_axis=1),
                               scalar=10, log_10=True)
            RA_cart = helper.toCartesianMask(RA, self.config_radar, gapfill_interval_num=interpolation)
            RA_img = helper.norm2Image(RA)[..., :3]
            RD_img = helper.norm2Image(RD)[..., :3]
            RA_cart_img = helper.norm2Image(RA_cart)[..., :3]

            img_file = loader.imgfileFromRADfile_new(RAD_filename, self.config_data["test_set_dir"], self.config_data["RAD_dir"])
            stereo_left_image = loader.readStereoLeft(img_file)

            RAD_data = helper.complexTo2Channels(RAD_complex)
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / self.config_data["global_variance_log"]

            gt_file = loader.gtfileFromRADfile_new(RAD_filename, self.config_data["test_set_dir"],
                                                   self.config_data["RAD_dir"])
            gt_instances = loader.readRadarInstances(gt_file)
            gt_labels, has_label, raw_boxes = self.encodeToLabels(gt_instances)
            gt_labels_cart, has_label_cart, raw_boxes_cart = self.encodeToCartBoxesLabels(gt_instances)

            data = torch.unsqueeze(torch.tensor(RAD_data, dtype=torch.float32), dim=0)
            label = torch.unsqueeze(torch.tensor(gt_labels[0], dtype=torch.float32), dim=0)
            label = torch.where(label == 0., torch.ones_like(label) * 1e-10, label)
            label_cart = torch.unsqueeze(torch.tensor(gt_labels_cart, dtype=torch.float32), dim=0)
            label_cart = torch.where(label_cart == 0., torch.ones_like(label_cart) * 1e-10, label_cart)
            raw_boxes = torch.unsqueeze(torch.tensor(raw_boxes, dtype=torch.float32), dim=0)
            raw_boxes_cart = torch.unsqueeze(torch.tensor(raw_boxes_cart, dtype=torch.float32), dim=0)
            # return RAD_filename, data, label, label_cart, raw_boxes, raw_boxes_cart, \
            #     stereo_left_image, RD_img, RA_img, RA_cart_img, gt_instances
            return data, label, label_cart, raw_boxes, raw_boxes_cart, \
                stereo_left_image, RD_img, RA_img, RA_cart_img, gt_instances


    def getGridStrides(self, ):
        """ Get grid strides """
        strides = (np.array(self.config_model["input_shape"])[:3] / np.array(self.headoutput_shape[1:4]))
        return np.array(strides).astype(np.float32)

    def getCartGridStrides(self, ):
        """ Get grid strides """
        if self.cart_shape is not None:
            cart_output_shape = [int(self.config_model["input_shape"][0]),
                                 int(2 * self.config_model["input_shape"][0])]
            strides = (np.array(cart_output_shape) / np.array(self.cart_shape[1:3]))
            return np.array(strides).astype(np.float32)
        else:
            return None

    def readSequences(self, mode):
        """ Read sequences from train/test directories. """
        assert mode in ["train", "test"]
        if mode == "train":
            sequences = glob.glob(os.path.join(self.config_data["train_set_dir"], "RAD/*/*.npy"))
        else:
            sequences = glob.glob(os.path.join(self.config_data["test_set_dir"], "RAD/*/*.npy"))
        if len(sequences) == 0:
            raise ValueError("Cannot read data from either train or test directory, "
                             "Please double-check the data path or the data format.")
        return sequences

    def splitTrain(self, train_sequences):
        """ Split train set to train and validate """
        total_num = len(train_sequences)
        validate_num = int(0.1 * total_num)
        if self.config_train["if_validate"]:
            return train_sequences[:total_num-validate_num], \
                train_sequences[total_num-validate_num:]
        else:
            return train_sequences, train_sequences[total_num-validate_num:]

    def encodeToLabels(self, gt_instances):
        """ Transfer ground truth instances into Detection Head format """
        raw_boxes_xyzwhd = np.zeros((self.config_data["max_boxes_per_frame"], 7))
        ### initialize gronud truth labels as np.zeors ###
        gt_labels = np.zeros(list(self.headoutput_shape[1:4]) + \
                             [len(self.anchor_boxes)] + \
                             [len(self.config_data["all_classes"]) + 7])

        ### start transferring box to ground turth label format ###
        for i in range(len(gt_instances["classes"])):
            if i > self.config_data["max_boxes_per_frame"]:
                continue
            class_name = gt_instances["classes"][i]
            box_xyzwhd = gt_instances["boxes"][i]
            class_id = self.config_data["all_classes"].index(class_name)
            if i < self.config_data["max_boxes_per_frame"]:
                raw_boxes_xyzwhd[i, :6] = box_xyzwhd
                raw_boxes_xyzwhd[i, 6] = class_id
            class_onehot = helper.smoothOnehot(class_id, len(self.config_data["all_classes"]))

            exist_positive = False

            grid_strid = self.grid_strides
            anchor_stage = self.anchor_boxes
            box_xyzwhd_scaled = box_xyzwhd[np.newaxis, :].astype(np.float32)
            box_xyzwhd_scaled[:, :3] /= grid_strid
            anchorstage_xyzwhd = np.zeros([len(anchor_stage), 6])
            anchorstage_xyzwhd[:, :3] = np.floor(box_xyzwhd_scaled[:, :3]) + 0.5
            anchorstage_xyzwhd[:, 3:] = anchor_stage.astype(np.float32)

            iou_scaled = helper.iou3d(box_xyzwhd_scaled, anchorstage_xyzwhd, \
                                      self.input_size)
            ### NOTE: 0.3 is from YOLOv4, maybe this should be different here ###
            ### it means, as long as iou is over 0.3 with an anchor, the anchor
            ### should be taken into consideration as a ground truth label
            iou_mask = iou_scaled > 0.3

            if np.any(iou_mask):
                xind, yind, zind = np.floor(np.squeeze(box_xyzwhd_scaled)[:3]). \
                    astype(np.int32)
                ### TODO: consider changing the box to raw yolohead output format ###
                gt_labels[xind, yind, zind, iou_mask, 0:6] = box_xyzwhd
                gt_labels[xind, yind, zind, iou_mask, 6:7] = 1.
                gt_labels[xind, yind, zind, iou_mask, 7:] = class_onehot
                exist_positive = True

            if not exist_positive:
                ### NOTE: this is the normal one ###
                ### it means take the anchor box with maximum iou to the raw
                ### box as the ground truth label
                anchor_ind = np.argmax(iou_scaled)
                xind, yind, zind = np.floor(np.squeeze(box_xyzwhd_scaled)[:3]). \
                    astype(np.int32)
                gt_labels[xind, yind, zind, anchor_ind, 0:6] = box_xyzwhd
                gt_labels[xind, yind, zind, anchor_ind, 6:7] = 1.
                gt_labels[xind, yind, zind, anchor_ind, 7:] = class_onehot

        has_label = False
        for label_stage in gt_labels:
            if label_stage.max() != 0:
                has_label = True
        gt_labels = [np.where(gt_i == 0, 1e-16, gt_i) for gt_i in gt_labels]
        return gt_labels, has_label, raw_boxes_xyzwhd

    def encodeToCartBoxesLabels(self, gt_instances):
        """ Transfer ground truth instances into Detection Head format """
        raw_boxes_xywh = np.zeros((self.config_data["max_boxes_per_frame"], 5))
        ### initialize gronud truth labels as np.zeros ###
        gt_labels = np.zeros(list(self.cart_shape[1:3]) + [len(self.anchor_boxes_cart)] +
                             [len(self.config_data["all_classes"]) + 5])
        ### start transferring box to ground turth label format ###
        for i in range(len(gt_instances["classes"])):
            if i > self.config_data["max_boxes_per_frame"]:
                continue
            class_name = gt_instances["classes"][i]
            box_xywh = gt_instances["cart_boxes"][i]
            class_id = self.config_data["all_classes"].index(class_name)
            if i <= self.config_data["max_boxes_per_frame"]:
                raw_boxes_xywh[i, :4] = box_xywh
                raw_boxes_xywh[i, 4] = class_id
            class_onehot = helper.smoothOnehot(class_id, len(self.config_data["all_classes"]))
            exist_positive = False
            grid_strid = self.cart_grid_strides
            anchors = self.anchor_boxes_cart
            box_xywh_scaled = box_xywh[np.newaxis, :].astype(np.float32)
            box_xywh_scaled[:, :2] /= grid_strid
            anchors_xywh = np.zeros([len(anchors), 4])
            anchors_xywh[:, :2] = np.floor(box_xywh_scaled[:, :2]) + 0.5
            anchors_xywh[:, 2:] = anchors.astype(np.float32)

            iou_scaled = helper.iou2d(box_xywh_scaled, anchors_xywh)
            ### NOTE: 0.3 is from YOLOv4, maybe this should be different here ###
            ### it means, as long as iou is over 0.3 with an anchor, the anchor
            ### should be taken into consideration as a ground truth label
            iou_mask = iou_scaled > 0.3

            if np.any(iou_mask):
                xind, yind = np.floor(np.squeeze(box_xywh_scaled)[:2]).astype(np.int32)
                ### TODO: consider changing the box to raw yolohead output format ###
                gt_labels[xind, yind, iou_mask, 0:4] = box_xywh
                gt_labels[xind, yind, iou_mask, 4:5] = 1.
                gt_labels[xind, yind, iou_mask, 5:] = class_onehot
                exist_positive = True

            if not exist_positive:
                ### NOTE: this is the normal one ###
                ### it means take the anchor box with maximum iou to the raw
                ### box as the ground truth label
                iou_mask = iou_scaled == iou_scaled.max()

                if np.any(iou_mask):
                    xind, yind = np.floor(np.squeeze(box_xywh_scaled)[:2]).astype(np.int32)
                    ### TODO: consider changing the box to raw yolohead output format ###
                    gt_labels[xind, yind, iou_mask, 0:4] = box_xywh
                    gt_labels[xind, yind, iou_mask, 4:5] = 1.
                    gt_labels[xind, yind, iou_mask, 5:] = class_onehot

        has_label = False
        if gt_labels.max() != 0:
            has_label = True
        gt_labels = np.where(gt_labels == 0, 1e-16, gt_labels)
        return gt_labels, has_label, raw_boxes_xywh