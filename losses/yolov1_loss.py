import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):

    def __init__(self, feature_size=7, num_bboxes=2, num_classes=20, lambda_coord=5.0, lambda_noobj=0.5):
        """ Constructor.
        Args:
            feature_size: (int) size of input feature map.
            num_bboxes: (int) number of bboxes per each cell.
            num_classes: (int) number of the object classes.
            lambda_coord: (float) weight for bbox location/size losses.
            lambda_noobj: (float) weight for no-objectness loss.
        """
        super(Loss, self).__init__()

        self.S = feature_size
        self.B = num_bboxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    @staticmethod
    def compute_iou(bbox1, bbox2):
        """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
        Args:
            bbox1: (Tensor) bounding bboxes, sized [N, 4].
            bbox2: (Tensor) bounding bboxes, sized [M, 4].
        Returns:
            (Tensor) IoU, sized [N, M].
        """
        N = bbox1.size(0)
        M = bbox2.size(0)

        # Compute left-top coordinate of the intersections
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand(N, M, 2)   # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Conpute right-bottom coordinate of the intersections
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)   # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute area of the intersections from the coordinates
        wh = rb - lt    # width and height of the intersection, [N, M, 2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

        # Compute area of the bboxes
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])  # [N, ]
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])  # [M, ]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M, ] -> [1, M] -> [N, M]

        # Compute IoU from the areas
        union = area1 + area2 - inter  # [N, M, 2]
        iou = inter / union            # [N, M, 2]

        return iou

    def forward(self, pred_tensor, target_tensor):
        """ Compute loss for YOLO training.
        Args:
            pred_tensor: (Tensor) predictions, sized [n_batch, S, S, Bx5+C], 5=len([x, y, w, h, conf]).
            target_tensor: (Tensor) targets, sized [n_batch, S, S, Bx5+C].
        Returns:
            (Tensor): loss, sized [1, ].
        """

        S, B, C = self.S, self.B, self.C
        N = 5 * B + C

        batch_size = pred_tensor.size(0)
        coord_mask = target_tensor[:, :, :, 4] > 0
        noobj_mask = target_tensor[:, :, :, 4] == 0
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target_tensor)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor)

        coord_pred = pred_tensor[coord_mask].view(-1, N)
        bbox_pred = coord_pred[:, :5 * B].contiguous().view(-1, 5)
        class_pred = coord_pred[:, 5 * B:]

        coord_target = target_tensor[coord_mask].view(-1, N)
        bbox_target = coord_target[:, :5 * B].contiguous().view(-1, 5)
        class_target = coord_target[:, 5 * B:]

        # Compute loss for the cells with no object bbox.
        noobj_pred = pred_tensor[noobj_mask].view(-1, N)
        noobj_target = target_tensor[noobj_mask].view(-1, N)
        noobj_conf_mask = torch.BoolTensor(noobj_pred.size()).fill_(0)
        for b in range(B):
            noobj_conf_mask[:, 4 + 5 * b] = 1
        noobj_pred_conf = noobj_pred[noobj_conf_mask]
        noobj_target_conf = noobj_target[noobj_conf_mask]
        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')

        # Compute loss for the cells with objects.
        coord_response_mask = torch.BoolTensor(bbox_target.size()).fill_(0)
        coord_not_response_mask = torch.BoolTensor(bbox_target.size()).fill_(1)
        bbox_target_iou = torch.zeros(bbox_target.size()).cuda()

        # Choose the predicted bbox having the highest IoU for each target bbox.
        for i in range(0, bbox_target.size(0), B):
            pred = bbox_pred[i:i + B]
            pred_xyxy = torch.FloatTensor(pred.size())
            pred_xyxy[:, :2] = pred[:, :2]/float(S) - 0.5 * pred[:, 2:4]
            pred_xyxy[:, 2:4] = pred[:, :2]/float(S) + 0.5 * pred[:, 2:4]

            target = bbox_target[i].view(-1, 5)
            target_xyxy = torch.FloatTensor(target.size())
            # Because (center_x,center_y) and (w,h) are normalized
            # for cell-size and image-size respectively, we rescale (center_x,center_y) for
            # the image-size to compute IoU correctly.
            target_xyxy[:, :2] = target[:, :2]/float(S) - 0.5 * target[:, 2:4]
            target_xyxy[:, 2:4] = target[:, :2]/float(S) + 0.5 * target[:, 2:4]

            iou = self.compute_iou(pred_xyxy[:, :4], target_xyxy[:, :4])
            max_iou, max_index = iou.max(0)
            max_index = max_index.data.cuda()

            coord_response_mask[i+max_index] = 1
            coord_not_response_mask[i+max_index] = 0

            # "we want the confidence score to equal the intersection over union (IOU) between the predicted box and
            # the ground truth" from the original paper of YOLO.
            bbox_target_iou[i + max_index, torch.LongTensor([4]).cuda()] = max_iou.data.cuda()
        bbox_target_iou = bbox_target_iou.cuda()

        # BBox location/size and objectness loss for the response bboxes.
        bbox_pred_response = bbox_pred[coord_response_mask].view(-1, 5)
        bbox_target_response = bbox_target[coord_response_mask].view(-1, 5)
        target_iou = bbox_target_iou[coord_response_mask].view(-1, 5)
        loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction='sum')
        loss_wh = F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]), torch.sqrt(bbox_target_response[:, 2:4]),
                             reduction='sum')
        loss_obj = F.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction='sum')

        # Class probability loss for the cells which contain objects.
        loss_class = F.mse_loss(class_pred, class_target, reduction='sum')

        # Total loss
        loss = self.lambda_coord * (loss_xy + loss_wh) + loss_obj + self.lambda_noobj * loss_noobj + loss_class
        loss = loss / float(batch_size)

        return loss
