import torch


class RadDetLoss(torch.nn.Module):
    def __init__(self, input_size, focal_loss_iou_threshold):
        super(RadDetLoss, self).__init__()
        self.input_size = input_size

        self.focal_loss_iou_thr = focal_loss_iou_threshold
        self.bce_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred_raw, pred, label, raw_boxes):
        assert len(raw_boxes.shape) == 3
        assert pred_raw.shape == label.shape
        assert pred_raw.shape[0] == len(raw_boxes)
        assert pred.shape == label.shape
        assert pred.shape[0] == len(raw_boxes)

        """
        TensorShape([3, 16, 16, 4, 6, 6]) TensorShape([3, 16, 16, 4, 6, 1]) TensorShape([3, 16, 16, 4, 6, 6])
        """
        raw_box, raw_conf, raw_category = extractYoloInfo(pred_raw)
        pred_box, pred_conf, pred_category = extractYoloInfo(pred)
        gt_box, gt_conf, gt_category = extractYoloInfo(label)

        giou_loss = self.yolo1Loss(pred_box, gt_box, gt_conf, self.input_size, if_box_loss_scale=False)
        focal_loss = self.focalLoss(raw_conf, pred_conf, gt_conf, pred_box, raw_boxes,
                                    self.input_size, self.focal_loss_iou_thr)
        category_loss = self.categoryLoss(raw_category, pred_category, gt_category, gt_conf)
        giou_total_loss = torch.mean(torch.sum(giou_loss, dim=[1, 2, 3, 4]))
        conf_total_loss = torch.mean(torch.sum(focal_loss, dim=[1, 2, 3, 4]))
        category_total_loss = torch.mean(torch.sum(category_loss, dim=[1, 2, 3, 4]))
        return giou_total_loss, conf_total_loss, category_total_loss

    def focalLoss(self, raw_conf, pred_conf, gt_conf, pred_box, raw_boxes, input_size, iou_loss_threshold=0.5):
        """
        Shpae of : iou, max_iou, gt_conf_negative, conf_focal, focal_loss
        TensorShape([3, 16, 16, 4, 6, 30]) TensorShape([3, 16, 16, 4, 6, 1]) TensorShape([3, 16, 16, 4, 6, 1]) TensorShape([3, 16, 16, 4, 6, 1]) TensorShape([3, 16, 16, 4, 6, 1])
        """
        """ Calculate focal loss for objectness """
        iou = tf_iou3d(torch.unsqueeze(pred_box, dim=-2), raw_boxes[:, None, None, None, None, :, :], input_size)
        max_iou = torch.unsqueeze(torch.max(iou, dim=-1)[0], dim=-1)

        gt_conf_negative = (1.0 - gt_conf) * (max_iou < iou_loss_threshold).to(torch.float32)
        conf_focal = torch.pow(gt_conf - pred_conf, 2.0)
        alpha = 0.01
        ###### TODO: think, whether we have to seperate logits with decoded outputs #######
        focal_loss = conf_focal * (gt_conf * self.bce_criterion(input=raw_conf, target=gt_conf) +
                                   alpha * gt_conf_negative * self.bce_criterion(input=raw_conf, target=gt_conf))
        return focal_loss

    def yolo1Loss(self, pred_box, gt_box, gt_conf, input_size, if_box_loss_scale=True):
        """ loss function for box regression \cite{YOLOV1} """
        assert pred_box.shape == gt_box.shape
        if if_box_loss_scale:
            scale = 2.0 - 1.0 * gt_box[..., 3:4] * gt_box[..., 4:5] * gt_box[..., 5:6] / \
                    (input_size[0] * input_size[1] * input_size[2])
        else:
            scale = 1.0
        ### NOTE: YOLOv1 original loss function ###
        giou_loss = gt_conf * scale * (torch.square(pred_box[..., :3] - gt_box[..., :3]) +
                                       torch.square(torch.sqrt(pred_box[..., 3:]) - torch.sqrt(gt_box[..., 3:])))
        return giou_loss

    def categoryLoss(self, raw_category, pred_category, gt_category, gt_conf):
        """ Category Cross Entropy loss """
        category_loss = gt_conf * self.bce_criterion(input=raw_category,
                                                     target=gt_category)
        return category_loss

def extractYoloInfo(yolo_output_format_data):
    """ Extract box, objectness, class from yolo output format data """
    box = yolo_output_format_data[..., :6]
    conf = yolo_output_format_data[..., 6:7]
    category = yolo_output_format_data[..., 7:]
    return box, conf, category


def tf_iou3d(box_xyzwhd_1, box_xyzwhd_2, input_size):
    """
    #############################################################################
    Shape of: box_xyzwhd_1, box_xyzwhd_2, input_size
    TensorShape([3, 16, 16, 4, 6, 1, 6]) TensorShape([3, 1, 1, 1, 1, 30, 6]) TensorShape([3]) [256 256 64]
    <class 'tensorflow.python.framework.ops.Tensor'> <class 'tensorflow.python.framework.ops.Tensor'> <class 'tensorflow.python.framework.ops.Tensor'>
    Shape of: box1_area, box2_area, box1_min, box1_max, box2_min, box2_max
    TensorShape([3, 16, 16, 4, 6, 1]) TensorShape([3, 1, 1, 1, 1, 30]) TensorShape([3, 16, 16, 4, 6, 1, 3]) TensorShape([3, 16, 16, 4, 6, 1, 3]) TensorShape([3, 1, 1, 1, 1, 30, 3]) TensorShape([3, 1, 1, 1, 1, 30, 3])
    Shape of: left_top, bottom_right, intersection, intersection_area, union_area, iou
    TensorShape([3, 16, 16, 4, 6, 30, 3]) TensorShape([3, 16, 16, 4, 6, 30, 3]) TensorShape([3, 16, 16, 4, 6, 30, 3]) TensorShape([3, 16, 16, 4, 6, 30]) TensorShape([3, 16, 16, 4, 6, 30]) TensorShape([3, 16, 16, 4, 6, 30])
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    """

    """ Tensorflow version of 3D bounding box IOU calculation
    Args:
        box_xyzwhd_1        ->      box1 [x, y, z, w, h, d]
        box_xyzwhd_2        ->      box2 [x, y, z, w, h, d]"""
    assert box_xyzwhd_1.shape[-1] == 6
    assert box_xyzwhd_2.shape[-1] == 6
    fft_shift_implement = torch.tensor([0, 0, input_size[2]/2]).to(box_xyzwhd_1.device)
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

    left_top = torch.maximum(box1_min, box2_min)
    bottom_right = torch.minimum(box1_max, box2_max)
    ### get intersection area
    intersection = torch.maximum(bottom_right - left_top,
                                 torch.zeros(bottom_right.shape, dtype=bottom_right.dtype, device=box_xyzwhd_1.device))
    intersection_area = intersection[..., 0] * intersection[..., 1] * intersection[..., 2]
    ### get union area
    union_area = box1_area + box2_area - intersection_area
    ### get iou
    iou = torch.nan_to_num(torch.div(intersection_area, union_area + 1e-10), 0.0)
    return iou




