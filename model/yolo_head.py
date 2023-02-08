import torch
import util.loader as loader
import torch.nn as nn
from einops import rearrange
import numpy as np
from model.yolo_loss import RadDetLoss


def iou3d(box_xyzwhd_1, box_xyzwhd_2, input_size):
    """ Numpy version of 3D bounding box IOU calculation
    Args:
        box_xyzwhd_1        ->      box1 [x, y, z, w, h, d]
        box_xyzwhd_2        ->      box2 [x, y, z, w, h, d]"""
    assert box_xyzwhd_1.shape[-1] == 6
    assert box_xyzwhd_2.shape[-1] == 6
    fft_shift_implement = np.array([0, 0, input_size[2]/2])
    ### areas of both boxes
    box1_area = box_xyzwhd_1[..., 3] * box_xyzwhd_1[..., 4] * box_xyzwhd_1[..., 5]
    box2_area = box_xyzwhd_2[..., 3] * box_xyzwhd_2[..., 4] * box_xyzwhd_2[..., 5]
    ### find the intersection box
    box1_min = box_xyzwhd_1[..., :3] + fft_shift_implement - box_xyzwhd_1[..., 3:] * 0.5
    box1_max = box_xyzwhd_1[..., :3] + fft_shift_implement + box_xyzwhd_1[..., 3:] * 0.5
    box2_min = box_xyzwhd_2[..., :3] + fft_shift_implement - box_xyzwhd_2[..., 3:] * 0.5
    box2_max = box_xyzwhd_2[..., :3] + fft_shift_implement + box_xyzwhd_2[..., 3:] * 0.5

    # box1_min = box_xyzwhd_1[..., :3] - box_xyzwhd_1[..., 3:] * 0.5
    # box1_max = box_xyzwhd_1[..., :3] + box_xyzwhd_1[..., 3:] * 0.5
    # box2_min = box_xyzwhd_2[..., :3] - box_xyzwhd_2[..., 3:] * 0.5
    # box2_max = box_xyzwhd_2[..., :3] + box_xyzwhd_2[..., 3:] * 0.5

    left_top = np.maximum(box1_min, box2_min)
    bottom_right = np.minimum(box1_max, box2_max)
    ### get intersection area
    intersection = np.maximum(bottom_right - left_top, 0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1] * intersection[..., 2]
    ### get union area
    union_area = box1_area + box2_area - intersection_area
    ### get iou
    iou = np.nan_to_num(intersection_area / (union_area + 1e-10))
    return iou

def nms(bboxes, iou_threshold, input_size, sigma=0.3, method='nms'):
    """ Bboxes format [x, y, z, w, h, d, score, class_index] """
    """ Implemented the same way as YOLOv4 """
    assert method in ['nms', 'soft-nms']
    if len(bboxes) == 0:
        best_bboxes = np.zeros([0, 8])
    else:
        all_pred_classes = list(set(bboxes[:, 7]))
        unique_classes = list(np.unique(all_pred_classes))
        best_bboxes = []
        for cls in unique_classes:
            cls_mask = (bboxes[:, 7] == cls)
            cls_bboxes = bboxes[cls_mask]
            ### NOTE: start looping over boxes to find the best one ###
            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 6])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                iou = iou3d(best_bbox[np.newaxis, :6], cls_bboxes[:, :6], \
                            input_size)
                weight = np.ones((len(iou),), dtype=np.float32)
                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0
                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))
                cls_bboxes[:, 6] = cls_bboxes[:, 6] * weight
                score_mask = cls_bboxes[:, 6] > 0.
                cls_bboxes = cls_bboxes[score_mask]
        if len(best_bboxes) != 0:
            best_bboxes = np.array(best_bboxes)
        else:
            best_bboxes = np.zeros([0, 8])
    return best_bboxes


def nmsOverClass(bboxes, iou_threshold, input_size, sigma=0.3, method='nms'):
    """ Bboxes format [x, y, z, w, h, d, score, class_index] """
    """ Implemented the same way as YOLOv4 """
    assert method in ['nms', 'soft-nms']
    if len(bboxes) == 0:
        best_bboxes = np.zeros([0, 8])
    else:
        best_bboxes = []
        ### NOTE: start looping over boxes to find the best one ###
        while len(bboxes) > 0:
            max_ind = np.argmax(bboxes[:, 6])
            best_bbox = bboxes[max_ind]
            best_bboxes.append(best_bbox)
            bboxes = np.concatenate([bboxes[: max_ind], bboxes[max_ind + 1:]])
            iou = iou3d(best_bbox[np.newaxis, :6], bboxes[:, :6], \
                        input_size)
            weight = np.ones((len(iou),), dtype=np.float32)
            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            bboxes[:, 6] = bboxes[:, 6] * weight
            score_mask = bboxes[:, 6] > 0.
            bboxes = bboxes[score_mask]
        if len(best_bboxes) != 0:
            best_bboxes = np.array(best_bboxes)
        else:
            best_bboxes = np.zeros([0, 8])
    return best_bboxes


def yoloheadToPredictions(yolohead_output, conf_threshold=0.5):
    """ Transfer YOLO HEAD output to [:, 8], where 8 means
    [x, y, z, w, h, d, score, class_index]"""
    prediction = yolohead_output.reshape(-1, yolohead_output.shape[-1])
    prediction_class = np.argmax(prediction[:, 7:], axis=-1)
    predictions = np.concatenate([prediction[:, :7], \
                                  np.expand_dims(prediction_class, axis=-1)], axis=-1)
    conf_mask = (predictions[:, 6] >= conf_threshold)
    predictions = predictions[conf_mask]
    return predictions


def decodeYolo(input, input_size, anchor_boxes, scale):
    x = rearrange(input, "n c d h w -> n h w c d")
    grid_size = x.shape[1:4]
    grid_strides = input_size / torch.tensor(list(grid_size), device=input.device)  # 16, 16, 16
    g0, g1, g2 = grid_size
    pred_raw = rearrange(x, "b h w a (c c1)-> b h w a c c1", c1=13)   # (None, 16, 16, 4, 78) ---> (None, 16, 16, 4, 6, 13)
    raw_xyz = pred_raw[:, :, :, :, :, :3]
    raw_whd = pred_raw[:, :, :, :, :, 3:6]
    raw_conf = pred_raw[:, :, :, :, :, 6:7]
    raw_prob = pred_raw[:, :, :, :, :, 7:]

    xx, yy, zz = torch.meshgrid([torch.arange(0, g0), torch.arange(0, g1), torch.arange(0, g2)])
    # xx = rearrange(xx, "h w c -> w h c")
    # yy = rearrange(yy, "h w c -> w h c")
    # zz = rearrange(zz, "h w c -> w h c")
    xyz_grid = [yy, xx, zz]
    xyz_grid = torch.stack(xyz_grid, dim=-1).to(input.device)
    xyz_grid = torch.unsqueeze(xyz_grid, 3)
    xyz_grid = rearrange(xyz_grid, "h w c a d -> w h c a d")
    xyz_grid = torch.unsqueeze(xyz_grid, 0)
    xyz_grid = torch.tile(xyz_grid, (input.shape[0], 1, 1, 1, len(anchor_boxes), 1))
    xyz_grid = xyz_grid.to(torch.float32)

    ### NOTE: not sure about this SCALE, but it appears in YOLOv4 tf version ###
    pred_xyz = ((torch.sigmoid(raw_xyz) * scale) - 0.5 * (scale - 1) + xyz_grid) * grid_strides
    ###---------------- clipping values --------------------###
    raw_whd = torch.clamp(raw_whd, 1e-12, 1e12)
    ###-----------------------------------------------------###
    pred_whd = torch.exp(raw_whd) * anchor_boxes
    pred_xyzwhd = torch.cat((pred_xyz, pred_whd), dim=-1)

    pred_conf = torch.sigmoid(raw_conf)
    pred_prob = torch.sigmoid(raw_prob)

    results = torch.cat((pred_xyzwhd, pred_conf, pred_prob), dim=-1)
    return pred_raw, results


class singleLayerHead(nn.Module):
    def __init__(self, num_anchors, num_class, last_channel, in_feature_size):
        super(singleLayerHead, self).__init__()
        self.num_anchor = num_anchors
        self.num_class = num_class
        self.last_channel = last_channel
        self.in_feature_size = in_feature_size

        final_output_channels = int(last_channel * self.num_anchor * (num_class + 7))  # 312
        self.final_output_reshape = [-1] + [int(last_channel),
                                            int(self.num_anchor) * (num_class + 7)] + list(in_feature_size[2:])

        self.conv1 = nn.Conv2d(in_channels=self.in_feature_size[1],
                               out_channels=self.in_feature_size[1]*2,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True,
                               )
        self.bn1 = nn.BatchNorm2d(self.in_feature_size[1]*2)

        self.conv2 = nn.Conv2d(in_channels=self.in_feature_size[1]*2,
                               out_channels=final_output_channels,
                               kernel_size=1,
                               stride=1,
                               bias=True
                               )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_feature):
        x = self.relu(self.bn1(self.conv1(input_feature)))
        x = self.conv2(x)
        x = x.view(self.final_output_reshape)
        return x


if __name__ == "__main__":
    input_features = torch.randn(3, 256, 16, 16)
    config = loader.readConfig(config_file_name="/home/albert_wei/WorkSpaces_2023/RADIA/RADDet/config.json")
    config_data = config["DATA"]
    config_radar = config["RADAR_CONFIGURATION"]
    config_model = config["MODEL"]
    config_train = config["TRAIN"]

    anchors_layer = anchor_boxes = loader.readAnchorBoxes(
        anchor_boxes_file="/home/albert_wei/WorkSpaces_2023/RADIA/RADDet/anchors.txt")  # load anchor boxes with order
    num_classes = len(config_data["all_classes"])

    input_size = list(config_model["input_shape"])
    input_channels = input_size[-1]
    num_class = len(config_data["all_classes"])
    yolohead_xyz_scales = config_model["yolohead_xyz_scales"]
    focal_loss_iou_threshold = config_train["focal_loss_iou_threshold"]

    # yolo_head = yoloHead(input_features, anchor_boxes, num_class)
    # output = yolo_head(input_features)
    input_tensor = torch.randn(3, 4, 78, 16, 16)
    pred_raw, pred = decodeYolo(input_tensor, input_size, anchor_boxes, yolohead_xyz_scales[0])

    data = torch.randn(3, 64, 256, 256)
    raw_boxes = torch.randn(3, 30, 7)
    label = torch.randn(3, 16, 16, 4, 6, 13)


    radarLoss = RadDetLoss(
        input_size=input_size,
        focal_loss_iou_threshold=focal_loss_iou_threshold
    )

    radarLoss(pred_raw=pred_raw,
              pred=pred,
              label=label,
              raw_boxes=raw_boxes[..., :6])
