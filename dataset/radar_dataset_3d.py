from torch.utils.data import Dataset
import numpy as np
import os, glob
import util.loader as loader
import util.helper as helper
from torchvision.transforms import ToTensor
import torch


class RararDataset3D(Dataset):
    def __init__(self, config_data, config_train, config_model, headoutput_shape,
                 anchors, transformer=ToTensor(), anchors_cart=None, cart_shape=None, dType="train"):
        super(RararDataset3D, self).__init__()
        self.input_size = config_model["input_shape"]
        self.config_data = config_data
        self.config_train = config_train
        self.config_model = config_model
        self.headoutput_shape = headoutput_shape
        self.cart_shape = cart_shape
        self.grid_strides = self.getGridStrides()
        self.cart_grid_strides = self.getCartGridStrides()
        self.anchor_boxes = anchors
        self.anchor_boxes_cart = anchors_cart
        self.RAD_sequences_train = self.readSequences(mode="train")
        self.RAD_sequences_test = self.readSequences(mode="test")
        ### NOTE: if "if_validat" set true in "config.json", it will split trainset ###
        self.RAD_sequences_train, self.RAD_sequences_validate = self.splitTrain(self.RAD_sequences_train)
        self.batch_size = config_train["batch_size"]
        self.total_train_batches = (self.config_train["epochs"] * len(self.RAD_sequences_train)) // self.batch_size
        self.total_test_batches = len(self.RAD_sequences_test) // self.batch_size
        self.total_validate_batches = len(self.RAD_sequences_validate) // self.batch_size
        self.dtype = dType
        self.transform = transformer

    def __len__(self):
        if self.dtype == "train":
            return len(self.RAD_sequences_train)
        elif self.dtype == "validate":
            return len(self.RAD_sequences_validate)
        elif self.dtype == "test":
            return len(self.RAD_sequences_test)
        else:
            raise ValueError("This type of dataset does not exist.")

    def __getitem__(self, index):
        if self.dtype == "train":
            return self.trainDataCart(index)
        elif self.dtype == "validate":
            return self.validateDataCart(index)
        elif self.dtype == "test":
            return self.testDataCart(index)
        else:
            raise ValueError("This type of dataset does not exist.")

    def trainDataCart(self, index):
        """ Generate train data with batch size """
        if self.cart_grid_strides is None:
            raise ValueError("Cartesian grid is None, please double check")

        has_label = False
        while not has_label:
            RAD_filename = self.RAD_sequences_train[index]
            RAD_complex = loader.readRAD(RAD_filename)
            if RAD_complex is None:
                raise ValueError("RAD file not found, please double check the path")
            ### NOTE: Gloabl Normalization ###
            RAD_data = helper.complexTo2Channels(RAD_complex)
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / self.config_data["global_variance_log"]
            ### load ground truth instances ###
            gt_filename = loader.gtfileFromRADfile(RAD_filename, self.config_data["train_set_dir"])
            gt_instances = loader.readRadarInstances(gt_filename)
            if gt_instances is None:
                raise ValueError("gt file not found, please double check the path")

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToCartBoxesLabels(gt_instances)
            index += 1
            gt_labels = np.stack(gt_labels, axis=0)
            if has_label:
                if self.transform:
                    return self.transform(RAD_data), torch.tensor(gt_labels, dtype=torch.float32), \
                        torch.tensor(raw_boxes, dtype=torch.float32)
                else:
                    return RAD_data, gt_labels, raw_boxes

    def testDataCart(self, index):
        if self.cart_grid_strides is None:
            raise ValueError("Cartesian grid is None, please double check")
        """ Generate test data with batch size """
        has_label = False
        while not has_label:
            RAD_filename = self.RAD_sequences_test[index]
            RAD_complex = loader.readRAD(RAD_filename)
            if RAD_complex is None:
                raise ValueError("RAD file not found, please double check the path")
            ### NOTE: Gloabl Normalization ###
            RAD_data = helper.complexTo2Channels(RAD_complex)
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / self.config_data["global_variance_log"]
            ### load ground truth instances ###
            gt_filename = loader.gtfileFromRADfile(RAD_filename, self.config_data["test_set_dir"])
            gt_instances = loader.readRadarInstances(gt_filename)
            if gt_instances is None:
                raise ValueError("gt file not found, please double check the path")

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToCartBoxesLabels(gt_instances)
            index += 1
            gt_labels = np.stack(gt_labels, axis=0)
            if has_label:
                if self.transform:
                    return self.transform(RAD_data), torch.tensor(gt_labels, dtype=torch.float32), \
                        torch.tensor(raw_boxes, dtype=torch.float32)
                else:
                    return RAD_data, gt_labels, raw_boxes

    def validateDataCart(self, index):
        if self.cart_grid_strides is None:
            raise ValueError("Cartesian grid is None, please double check")
        """ Generate test data with batch size """
        has_label = False
        while not has_label:
            RAD_filename = self.RAD_sequences_validate[index]
            RAD_complex = loader.readRAD(RAD_filename)
            if RAD_complex is None:
                raise ValueError("RAD file not found, please double check the path")
            ### NOTE: Gloabl Normalization ###
            RAD_data = helper.complexTo2Channels(RAD_complex)
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / self.config_data["global_variance_log"]
            ### load ground truth instances ###
            gt_filename = loader.gtfileFromRADfile(RAD_filename, self.config_data["train_set_dir"])
            gt_instances = loader.readRadarInstances(gt_filename)
            if gt_instances is None:
                raise ValueError("gt file not found, please double check the path")
            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToCartBoxesLabels(gt_instances)
            index += 1
            gt_labels = np.stack(gt_labels, axis=0)
            if has_label:
                if self.transform:
                    return self.transform(RAD_data), torch.tensor(gt_labels, dtype=torch.float32), \
                        torch.tensor(raw_boxes, dtype=torch.float32)
                else:
                    return RAD_data, gt_labels, raw_boxes

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

    def getGridStrides(self, ):
        """ Get grid strides """
        strides = (np.array(self.config_model["input_shape"])[:3] / np.array(self.headoutput_shape[1:4]))
        return np.array(strides).astype(np.float32)

    def getCartGridStrides(self, ):
        """ Get grid strides """
        if self.cart_shape is not None:
            cart_output_shape = [int(self.config_model["input_shape"][0]), int(2 * self.config_model["input_shape"][0])]
            strides = (np.array(cart_output_shape) / np.array(self.cart_shape[1:3]))
            return np.array(strides).astype(np.float32)
        else:
            return None

    def readSequences(self, mode):
        """ Read sequences from train/test directories. """
        assert mode in ["train", "test"]
        if mode == "train":
            sequences = glob.glob(os.path.join(self.config_data["train_set_dir"], "RAD/*/*.npy"))
        elif mode == "test":
            sequences = glob.glob(os.path.join(self.config_data["test_set_dir"], "RAD/*/*.npy"))
        else:
            raise ValueError(f"{mode} type does not exist.")
        if len(sequences) == 0:
            raise ValueError("Cannot read data from either train or test directory, \
                        Please double-check the data path or the data format.")
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