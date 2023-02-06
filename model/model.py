import numpy as np
import torch.nn as nn
import torchvision
from model.backbone_radarResNet import build_backbone
from model.yolo_head import singleLayerHead, decodeYolo


class RADDet(nn.Module):
    def __init__(self, config_model, config_data, config_train, anchor_boxes):
        super(RADDet, self).__init__()
        assert (isinstance(config_model["input_shape"], tuple) or isinstance(config_model["input_shape"], list))
        self.input_size = list(config_model["input_shape"])
        self.input_channels = self.input_size[-1]
        self.num_class = len(config_data["all_classes"])
        self.anchor_boxes = anchor_boxes
        self.yolohead_xyz_scales = config_model["yolohead_xyz_scales"]
        self.focal_loss_iou_threshold = config_train["focal_loss_iou_threshold"]
        self.yolo_feature_size = config_model["yolo_feature_size"]
        self.backbone = build_backbone(config_model)
        self.yolo_head = singleLayerHead(num_anchors=len(self.anchor_boxes),
                                         num_class=self.num_class,
                                         last_channel=int(self.yolo_feature_size[-1]/4),
                                         in_feature_size=self.yolo_feature_size)
        # self.radarLoss = RadDetLoss(
        #     input_size=self.input_size,
        #     focal_loss_iou_threshold=self.focal_loss_iou_threshold
        # )

    def forward(self, input):
        out = self.backbone(input)
        out = self.yolo_head(out)
        return out