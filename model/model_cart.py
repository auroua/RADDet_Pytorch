import torch
import torch.nn as nn
from einops import rearrange
import util.loader as loader
from model.model import RADDet
from dataset.radar_dataset_3d import RararDataset3D


class RADDetCart(nn.Module):
    def __init__(self, config_model, config_data, config_train, anchor_boxes, input_shape, device, backbone=None):
        """ make sure the model is buit when initializint the class.
        Only by this, the graph could be built and the trainable_variables
        could be initialized """
        super(RADDetCart, self).__init__()
        assert (isinstance(input_shape, tuple) or isinstance(input_shape, list))
        self.config_model = config_model
        self.config_data = config_data
        self.config_train = config_train
        self.input_size = input_shape
        self.num_class = len(config_data["all_classes"])
        self.anchor_boxes = torch.tensor(anchor_boxes).to(device)
        self.yolohead_xyz_scales = config_model["yolohead_xyz_scales"]
        self.focal_loss_iou_threshold = config_train["focal_loss_iou_threshold"]

        self.bce_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.backbone = backbone

        dense_feature_size = input_shape[0]*input_shape[1]
        self.fc1 = nn.Linear(in_features=dense_feature_size, out_features=dense_feature_size*2, bias=True)
        self.fc2 = nn.Linear(in_features=dense_feature_size*2, out_features=dense_feature_size*2, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels=self.input_size[-1],
                               out_channels=self.input_size[-1],
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True
                               )
        self.bn1 = nn.BatchNorm2d(self.input_size[-1])

        self.conv2 = nn.Conv2d(in_channels=self.input_size[-1],
                               out_channels=self.input_size[-1],
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True
                               )
        self.bn2 = nn.BatchNorm2d(self.input_size[-1])

        self.conv3 = nn.Conv2d(in_channels=self.input_size[-1],
                               out_channels=self.input_size[-1],
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True
                               )
        self.bn3 = nn.BatchNorm2d(self.input_size[-1])

        self.conv_yolo_head = nn.Conv2d(in_channels=self.input_size[-1],
                                        out_channels=self.input_size[-1]*2,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=True
                                        )
        self.bn_yolo_head = nn.BatchNorm2d(self.input_size[-1]*2)

        self.conv_yolo_head_1 = nn.Conv2d(in_channels=self.input_size[-1]*2,
                                          out_channels=len(self.anchor_boxes) * (self.num_class + 5),
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          bias=True
                                          )

        # TODO: adding weight initialization code.
    def forward(self, x):
        x = self.backbone(x)
        x = rearrange(x, "b c h w -> b c (h w)")
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = rearrange(x, "b c (h w) -> b c h w", h=self.input_size[0])
        res_x = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x + res_x

        x = self.relu(self.bn_yolo_head(self.conv_yolo_head(x)))
        x = self.conv_yolo_head_1(x)
        x = rearrange(x, "b c h w -> b h w c")
        x = rearrange(x, "b h w (c1 c2) -> b h w c1 c2", c1=len(self.anchor_boxes))
        return x

    def decodeYolo(self, x):
        output_size = [int(self.config_model["input_shape"][0]), int(2 * self.config_model["input_shape"][0])]
        strides = torch.tensor(output_size) / torch.tensor(list(x.shape[1:3]))
        strides = strides.to(x.device)
        raw_xy, raw_wh, raw_conf, raw_prob = x[..., 0:2], x[..., 2:4], x[..., 4:5], x[..., 5:]
        xx, yy = torch.meshgrid([torch.arange(0, x.shape[1]), torch.arange(0, x.shape[2])])
        xy_grid = [xx.T, yy.T]
        xy_grid = torch.unsqueeze(torch.stack(xy_grid, dim=-1), dim=-2).to(x.device)
        xy_grid = torch.unsqueeze(rearrange(xy_grid, "b c h w -> c b h w"), dim=0)
        xy_grid = torch.tile(xy_grid, (x.shape[0], 1, 1, len(self.anchor_boxes), 1)).to(torch.float32)
        scale = self.yolohead_xyz_scales[0]
        pred_xy = ((torch.sigmoid(raw_xy) * scale) - 0.5 * (scale - 1) + xy_grid) * strides
        ###---------------- clipping values --------------------###
        raw_wh = torch.clamp(raw_wh, 1e-12, 1e12)
        ###-----------------------------------------------------###
        pred_wh = torch.exp(raw_wh) * self.anchor_boxes
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)

        pred_conf = torch.sigmoid(raw_conf)
        pred_prob = torch.sigmoid(raw_prob)
        return torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)

    def extractYoloInfo(self, yoloformat_data):
        box = yoloformat_data[..., :4]
        conf = yoloformat_data[..., 4:5]
        category = yoloformat_data[..., 5:]
        return box, conf, category

    def loss(self, pred_raw, pred, gt, raw_boxes):
        raw_box, raw_conf, raw_category = self.extractYoloInfo(pred_raw)
        pred_box, pred_conf, pred_category = self.extractYoloInfo(pred)
        gt_box, gt_conf, gt_category = self.extractYoloInfo(gt)

        box_loss = gt_conf * (torch.square(pred_box[..., :2] - gt_box[..., :2]) +
                              torch.square(torch.sqrt(pred_box[..., 2:]) - torch.sqrt(gt_box[..., 2:])))
        iou = tf_iou2d(torch.unsqueeze(pred_box, dim=-2), raw_boxes[:, None, None, None, :, :])
        max_iou = torch.unsqueeze(torch.max(iou, dim=-1)[0], dim=-1)
        gt_conf_negative = (1.0 - gt_conf) * (max_iou < self.config_train["focal_loss_iou_threshold"]).to(torch.float32)
        conf_focal = torch.pow(gt_conf - pred_conf, 2)
        alpha = 0.01

        conf_loss = conf_focal * (gt_conf * self.bce_criterion(target=gt_conf, input=raw_conf)
                                  + alpha * gt_conf_negative * self.bce_criterion(target=gt_conf, input=raw_conf))
        ### NOTE: category loss function ###
        category_loss = gt_conf * self.bce_criterion(target=gt_category, input=raw_category)

        ### NOTE: combine together ###
        box_loss_all = torch.mean(torch.sum(box_loss, dim=[1, 2, 3, 4]))
        box_loss_all *= 1e-1
        conf_loss_all = torch.mean(torch.sum(conf_loss, dim=[1, 2, 3, 4]))
        category_loss_all = torch.mean(torch.sum(category_loss, dim=[1, 2, 3, 4]))
        total_loss = box_loss_all + conf_loss_all + category_loss_all
        return total_loss, box_loss_all, conf_loss_all, category_loss_all


def tf_iou2d(box_xywh_1, box_xywh_2):
    """ Tensorflow version of 3D bounding box IOU calculation
    Args:
        box_xywh_1        ->      box1 [x, y, w, h]
        box_xywh_2        ->      box2 [x, y, w, h]"""
    assert box_xywh_1.shape[-1] == 4
    assert box_xywh_2.shape[-1] == 4
    ### areas of both boxes
    box1_area = box_xywh_1[..., 2] * box_xywh_1[..., 3]
    box2_area = box_xywh_2[..., 2] * box_xywh_2[..., 3]
    ### find the intersection box
    box1_min = box_xywh_1[..., :2] - box_xywh_1[..., 2:] * 0.5
    box1_max = box_xywh_1[..., :2] + box_xywh_1[..., 2:] * 0.5
    box2_min = box_xywh_2[..., :2] - box_xywh_2[..., 2:] * 0.5
    box2_max = box_xywh_2[..., :2] + box_xywh_2[..., 2:] * 0.5

    left_top = torch.maximum(box1_min, box2_min)
    bottom_right = torch.minimum(box1_max, box2_max)
    ### get intersection area
    intersection = torch.maximum(bottom_right - left_top,
                                 torch.zeros(bottom_right.shape, dtype=bottom_right.dtype, device=box_xywh_1.device))
    intersection_area = intersection[..., 0] * intersection[..., 1]
    ### get union area
    union_area = box1_area + box2_area - intersection_area
    ### get iou
    iou = torch.nan_to_num(torch.div(intersection_area, union_area + 1e-10), 0.0)
    return iou


if __name__ == "__main__":
    input_features = torch.randn(3, 256, 16, 16)
    config = loader.readConfig(config_file_name="/home/albert_wei/WorkSpaces_2023/RADIA/RADDet/config.json")
    config_data = config["DATA"]
    config_radar = config["RADAR_CONFIGURATION"]
    config_model = config["MODEL"]
    config_train = config["TRAIN"]

    anchors_layer = anchor_boxes = loader.readAnchorBoxes(
        anchor_boxes_file="/home/albert_wei/WorkSpaces_2023/RADIA/RADDet/anchors.txt")

    anchors_layer_cart = anchor_boxes_cart = loader.readAnchorBoxes(
        anchor_boxes_file="/home/albert_wei/WorkSpaces_2023/RADIA/RADDet/anchors_cartboxes.txt")  # load anchor boxes with order
    num_classes = len(config_data["all_classes"])

    input_size = list(config_model["input_shape"])
    input_channels = input_size[-1]
    num_class = len(config_data["all_classes"])
    yolohead_xyz_scales = config_model["yolohead_xyz_scales"]
    focal_loss_iou_threshold = config_train["focal_loss_iou_threshold"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input = torch.randn(3, 64, 256, 256)
    input = input.to(device)

    model = RADDet(config_model, config_data, config_train, anchor_boxes_cart)
    model.to(device)
    bk = model.backbone

    cart_model = RADDetCart(config_model=config_model,
                            config_data=config_data,
                            config_train=config_train,
                            anchor_boxes=anchor_boxes_cart,
                            input_shape=config_model["bk_output_size"],
                            device=device)
    cart_model.to(device)

    out = bk(input)
    pred_raw = cart_model(out)
    pred = cart_model.decodeYolo(pred_raw)

    label = torch.randn(3, 16, 32, 6, 11).to(device)
    raw_box = torch.randn(3, 30, 5).to(device)

    cart_model.loss(pred_raw, pred, label, raw_box[..., :4])

    radar_dataset_3d = RararDataset3D(config_data=config_data,
                                      config_train=config_train,
                                      config_model=config_model,
                                      headoutput_shape=[3, 16, 16, 4, 78],
                                      anchors=anchor_boxes,
                                      anchors_cart=anchor_boxes_cart,
                                      cart_shape=[3, 16, 32, 6, 11])
    for d in radar_dataset_3d:
        data, gt_label, raw_boxes = d
        print(data.shape, gt_label.shape, raw_boxes.shape)

