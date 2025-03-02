# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors,make_anchors_polar

from .metrics import bbox_iou
from .tal import bbox2dist
import cv2


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367."""

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') *
                    weight).mean(1).sum()
        return loss

class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self, ):
        super().__init__()

    def forward(self, pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()

class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)

        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)

# class MaskIOULoss(nn.Module):
#     def __init__(self):
#         super(MaskIOULoss, self).__init__()

#     def forward(self, pred_rays, target_rays, centerness_gt):
#         '''
#          :param pred_rays:  shape (N,36), N is nr_box
#          :param target: shape (N,36)
#          :return: loss
#          '''
        
#         num=centerness_gt.sum()
#         total = torch.stack([pred_rays,target_rays], -1)# n 36 2 
#         l_max = total.max(dim=2)[0]
#         l_min = total.min(dim=2)[0].clamp(min=1e-6)
#         loss = (l_max.sum(dim=1) / l_min.sum(dim=1)).log()
#         loss = loss * centerness_gt
#         loss = loss.sum() / num
#         return loss

class MaskIOULoss(nn.Module):
    def __init__(self):
        super(MaskIOULoss, self).__init__()

    def forward(self, pred_rays, target_rays,target_scores,target_scores_sum):
        '''
         :param pred_rays:  shape (N,36), N is nr_box
         :param target: shape (N,36)
         :return: loss
         '''

        weight = target_scores.sum(-1)# num,1
        total = torch.stack([pred_rays,target_rays], -1)# n 36 2 
        l_max = total.max(dim=2)[0]
        l_min = total.min(dim=2)[0].clamp(min=1e-6)
        loss = (l_max.sum(dim=1) / l_min.sum(dim=1)).log()
        loss = loss * weight
        loss = loss.sum() / target_scores_sum
        return loss

# class MaskIOULoss(nn.Module):
#     def __init__(self):
#         super(MaskIOULoss, self).__init__()

#     def forward(self, pred_rays, target_rays,fg_mask, centerness_gt,target_scores_sum=1,target_scores=1,weight=1):
#         '''
#          :param pred_rays:  shape (N,36), N is nr_box
#          :param target: shape (N,36)
#          :return: loss
#          '''
#         # weight = target_scores.sum(-1)[fg_mask]       
#         # num=fg_mask.sum()
#         num=centerness_gt.sum()

#         total = torch.stack([pred_rays,target_rays], -1)# n 36 2 
#         l_max = total.max(dim=2)[0].clamp(min=1e-6)
#         l_min = total.min(dim=2)[0].clamp(min=1e-6)
#         loss = 1-(l_min.sum(dim=1) / l_max.sum(dim=1))
#         # print(num)
#         loss = loss *centerness_gt
#         loss = loss.sum() / num
#         # loss = loss.sum() / num
#         return loss


class centernessloss(nn.Module):
    def __init__(self):
        super(centernessloss, self).__init__()

    def forward(self, center_pred,fg_mask,centerness_gt=1):
        '''
         :param pred_rays:  shape (N,36), N is nr_box
         :param target: shape (N,36)
         :return: loss
         '''
        num=fg_mask.sum()
        loss = F.binary_cross_entropy_with_logits(center_pred, centerness_gt.float(),reduction='none')
        loss = loss.sum() / num
        return loss




class KeypointLoss(nn.Module):

    def __init__(self, sigmas) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0)) / (torch.sum(kpt_mask != 0) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2 * self.sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        return kpt_loss_factor * ((1 - torch.exp(-e)) * kpt_mask).mean()







class v8polarpaperv8DetectionLoss:
    #v8polarpaper
    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=4.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.polar_loss = MaskIOULoss()
        self.center_loss=centernessloss()
    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        # print(targets.shape)
        if targets.shape[0] == 0: # targets mat shape is [n,6] n is instance num in batch 
            out = torch.zeros(batch_size, 0, 4+1+360*2, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            # print(i)
            # print(counts)
            # print(counts.max())
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 4+1+360*2, device=self.device) # b, instance_num, num
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:#if image exit instance, get all n instance , in addition to index behind n
                    out[j, :n] = targets[matches, 1:]

            ## if image not exit any instance, correspondingly out is all 0 in certain index(dim=0)
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
            out[..., 5:5+360]=out[..., 5:5+360]*scale_tensor[0]
            out[..., 5+360:]=out[..., 5+360:]*scale_tensor[1]
            #极坐标没有实际值的地方为0
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # ray, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (36, self.nc+1), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()#b,anchor,c+1
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()#b,anchor,36

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)#instance_num,num
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])## b, instance_num, num
        # print(targets)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)#计算box坐标累加大于0的为True

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)






class polarv8DetectionLoss:
    #polarLoss
    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=4.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.polar_loss = MaskIOULoss()
        self.center_loss=centernessloss()
    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        # print(targets.shape)
        if targets.shape[0] == 0: # targets mat shape is [n,6] n is instance num in batch 
            out = torch.zeros(batch_size, 0, 4+1+360*2, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            # print(i)
            # print(counts)
            # print(counts.max())
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 4+1+360*2, device=self.device) # b, instance_num, num
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:#if image exit instance, get all n instance , in addition to index behind n
                    out[j, :n] = targets[matches, 1:]

            ## if image not exit any instance, correspondingly out is all 0 in certain index(dim=0)
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
            out[..., 5:5+360]=out[..., 5:5+360]*scale_tensor[0]
            out[..., 5+360:]=out[..., 5+360:]*scale_tensor[1]
            #极坐标没有实际值的地方为0
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # ray, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (36, self.nc+1), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()#b,anchor,c+1
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()#b,anchor,36

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        # print('anchor_points',anchor_points)
        # print('anchor_points',anchor_points.shape)


        # print(batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes'])
        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)#instance_num,num
        # targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['masks']), 1)

        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])## b, instance_num, num
        # print(targets)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)#计算box坐标累加大于0的为True

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

# Criterion class for computing Detection training losses
class oriv8DetectionLoss:

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

class v8DetectionLoss:
    #nofpn 1head
    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no+32
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.stride_proj = torch.arange(32, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)


    def stride_decode(self, pred_stride):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_stride.shape  # batch, anchors, channels
            pred_stride = pred_stride.view(b, a, 1, c).softmax(3).matmul(self.stride_proj.type(pred_stride.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return pred_stride

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores,pred_stride = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc,self.reg_max*2), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_stride = pred_stride.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_stride=self.stride_decode(pred_stride) # xyxy, (b, h*w, 1)
        pred_bboxes=pred_bboxes*pred_stride
        _, target_bboxes, target_scores, fg_mask, _ ,target_strides= self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach()).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= target_strides
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

class v8SegmentationLoss(v8DetectionLoss):
    #polar
    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.nm = model.model[-1].nm  # number of masks
        self.overlap = model.args.overlap_mask


    def get_centerpoint(self,liss):
        
        xycenter=torch.zeros((liss.shape[0],liss.shape[1],2), device=self.device)
        for j in range(liss.shape[0]):
            for k in range(liss.shape[1]):
                lis=liss[j,k].view(360,2)
                area = 0.0
                x, y = 0.0, 0.0
                a = len(lis)
                for i in range(a):
                    lat = lis[i,0]
                    lng = lis[i,1]
                    if i == 0:
                        lat1 = lis[-1,0]
                        lng1 = lis[-1,1]
                    else:
                        lat1 = lis[i - 1,0]
                        lng1 = lis[i - 1,1]
                    fg = (lat * lng1 - lng * lat1) / 2.0
                    area += fg
                    x += fg * (lat + lat1) / 3.0
                    y += fg * (lng + lng1) / 3.0
                x = x / area
                y = y / area
                xycenter[j,k,0]=x
                xycenter[j,k,1]=y
        return xycenter

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(2, device=self.device)  # box, cls, dfl

        feats,_,_ = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = feats[0].shape  # batch size, number of masks, mask height, mask width

        # print(feats[0].shape)
        # print(pred_masks.shape)

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (36, self.nc), 1)
        # print(pred_distri.shape)
        # print(pred_scores.shape)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)

        anchor_points, stride_tensor,ss = make_anchors_polar(feats, self.stride, 0.5)
        # targets
        try:
            batch_idx = batch['batch_idx'].view(-1, 1)
            import numpy as np
            seg=torch.cat(batch['segments']).contiguous().view(-1,360*2).to(batch_idx.device)

            targets = torch.cat((batch_idx, batch['cls'].contiguous().view(-1, 1), batch['bboxes'],seg), 1)#batch['bboxes']  n,4
            # print(targets.shape)

            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            # print('max_num_objmtargetss',targets.shape)


            gt_labels, gt_bboxes,gt_coor = targets.split((1, 4,360*2), 2)  # cls, xyxy
            # gt_center=self.get_centerpoint(gt_coor)#重心
            gt_center=0
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        except RuntimeError as e:


            # print(batch)
           
            # print(len(batch['segments']))
            # print(batch['segments'][0].shape)
            # print(torch.cat(batch['segments']))
            # print(torch.cat(batch['segments']).shape)
            # print(seg)
            raise TypeError('ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n'
                            "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                            "i.e. 'yolo train model=yolov8n-seg.pt data=coco128.yaml'.\nVerify your dataset is a "
                            "correctly formatted 'segment' dataset using 'data=coco128-seg.yaml' "
                            'as an example.\nSee https://docs.ultralytics.com/tasks/segment/ for help.') from e

        # pboxes
        # pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # print(stride_tensor.shape)
        # pred_distri=F.relu(pred_distri)
        # pred_distri=pred_distri.exp()*stride_tensor.clamp(min=1e-6)
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx,target_dist,centerness_gt = self.assigner(
            pred_scores.detach().sigmoid(), (pred_distri.detach()*stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt,gt_coor,stride_tensor,ss,gt_center)

        target_scores_sum = max(target_scores.sum(), 1)

        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        if fg_mask.sum():
            # print(l)
            # print(pred_distri.shape)
            # print(target_bboxes[fg_mask].shape)
            # print(target_bboxes.shape)
            # print(target_dist.shape)
            # print(target_dist)
            # print(l)
            # print(stride_tensor[fg_mask].shape)
            # masks loss
            loss[0]=self.polar_loss(pred_distri[fg_mask], target_dist,fg_mask,target_scores_sum,target_scores)
        loss[0] *= self.hyp.box  # box gain
        # loss[0] *= 3  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)



class polarv8SegmentationLoss(v8DetectionLoss):
    #polar 回归中心点
    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.nm = model.model[-1].nm  # number of masks
        self.overlap = model.args.overlap_mask


    def get_centerpoint(self,liss):
        
        xycenter=torch.zeros((liss.shape[0],liss.shape[1],2), device=self.device)
        for j in range(liss.shape[0]):
            for k in range(liss.shape[1]):
                lis=liss[j,k].view(360,2)
                area = 0.0
                x, y = 0.0, 0.0
                a = len(lis)
                for i in range(a):
                    lat = lis[i,0]
                    lng = lis[i,1]
                    if i == 0:
                        lat1 = lis[-1,0]
                        lng1 = lis[-1,1]
                    else:
                        lat1 = lis[i - 1,0]
                        lng1 = lis[i - 1,1]
                    fg = (lat * lng1 - lng * lat1) / 2.0
                    area += fg
                    x += fg * (lat + lat1) / 3.0
                    y += fg * (lng + lng1) / 3.0
                x = x / area
                y = y / area
                xycenter[j,k,0]=x
                xycenter[j,k,1]=y
        return xycenter

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl

        feats,_,_ = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = feats[0].shape  # batch size, number of masks, mask height, mask width

        # print(feats[0].shape)
        # print(pred_masks.shape)
        # print(self.nc)
        # print(feats[0].shape)

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.nm, self.nc), 1)
        
        new_pred_distri=pred_distri[:,:36,:]
        pred_centerness=pred_distri[:,36:,:]

        # print(new_pred_distri.shape)
        # print(pred_scores.shape)
        # print(pred_centerness.shape)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        new_pred_distri = new_pred_distri.permute(0, 2, 1).contiguous()
        pred_centerness = pred_centerness.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)

        anchor_points, stride_tensor,ss = make_anchors_polar(feats, self.stride, 0.5)
        # targets
        try:
            batch_idx = batch['batch_idx'].view(-1, 1)
            import numpy as np


            seg=torch.cat(batch['segments']).contiguous().view(-1,360*2).to(batch_idx.device)

            targets = torch.cat((batch_idx, batch['cls'].contiguous().view(-1, 1), batch['bboxes'],seg), 1)#batch['bboxes']  n,4  
            # print(targets.shape)

            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])


            gt_labels, gt_bboxes,gt_coor = targets.split((1, 4,360*2), 2)  # cls, xyxy
            gt_center=0
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
            oriimg=batch['img']*255
            oriimg = oriimg.permute(0,2,3,1).cpu().detach().numpy()
            # cv2.imwrite('/home/data2/mxl/ultralytics/ultralytics-main/output_high_quality.jpg', oriimg[0])
            # print(seg.shape)
            # print(torch.cat(batch['segments']).shape)
            # print(batch['batch_idx'].shape)


        except RuntimeError as e:

            # print(batch)
           
            # print(len(batch['segments']))
            # print(batch['segments'][0].shape)
            # print(torch.cat(batch['segments']))
            # print(torch.cat(batch['segments']).shape)
            # print(seg)
            raise TypeError('ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n'
                            "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                            "i.e. 'yolo train model=yolov8n-seg.pt data=coco128.yaml'.\nVerify your dataset is a "
                            "correctly formatted 'segment' dataset using 'data=coco128-seg.yaml' "
                            'as an example.\nSee https://docs.ultralytics.com/tasks/segment/ for help.') from e
        # pboxes
        new_pred_distri=new_pred_distri*(stride_tensor.detach())
        n_max_boxes = gt_bboxes.size(1)
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx,target_dist,centerness_gt,fg_mask222 = self.assigner(
            pred_scores.detach().sigmoid(), (new_pred_distri.detach()).type(gt_bboxes.dtype),
            (anchor_points * stride_tensor).detach(), gt_labels, gt_bboxes, mask_gt,gt_coor,stride_tensor,ss,gt_center,oriimg,imgsz)
        
        one_hot_vector = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).type(gt_bboxes.dtype).to(gt_bboxes.device)

        target_scores = torch.where(~fg_mask222.unsqueeze(-1).expand_as(target_scores), one_hot_vector, target_scores)  
        target_scores_sum = max(target_scores.sum(), 1)
        # loss[2]=self.bce(pred_centerness,fg_mask.unsqueeze(-1).to(dtype)).sum()/fg_mask.sum()
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / fg_mask222.sum()  # BCE
        # print(target_scores_sum)
        # print(pred_scores.shape)
        # print(target_scores.shape)
        # print(fg_mask.shape)
        # print(new_pred_distri.shape)
        # print(l)
        if fg_mask.sum():
            # masks loss
            # loss[1] = self.bce(pred_scores[fg_mask], target_scores[fg_mask].to(dtype)).sum() / fg_mask.sum()  # BCE
            new_pred_distri = new_pred_distri.unsqueeze(1).expand(-1, n_max_boxes,-1, -1)[fg_mask.bool()]
            pred_centerness = pred_centerness.unsqueeze(1).expand(-1, n_max_boxes,-1, -1)[fg_mask.bool()]

            loss[0]=self.polar_loss(new_pred_distri,target_dist,centerness_gt)
            loss[2]=self.center_loss(pred_centerness.squeeze(-1),fg_mask,centerness_gt)
            

        # loss[0] *= self.hyp.box  # box gain
        loss[0] *= 2  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)



class v8SegmentationLoss(v8DetectionLoss):
    #polarpaper 
    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.nm = model.model[-1].nm  # number of masks
        self.overlap = model.args.overlap_mask


    def get_centerpoint(self,liss):
        
        xycenter=torch.zeros((liss.shape[0],liss.shape[1],2), device=self.device)
        for j in range(liss.shape[0]):
            for k in range(liss.shape[1]):
                lis=liss[j,k].view(360,2)
                area = 0.0
                x, y = 0.0, 0.0
                a = len(lis)
                for i in range(a):
                    lat = lis[i,0]
                    lng = lis[i,1]
                    if i == 0:
                        lat1 = lis[-1,0]
                        lng1 = lis[-1,1]
                    else:
                        lat1 = lis[i - 1,0]
                        lng1 = lis[i - 1,1]
                    fg = (lat * lng1 - lng * lat1) / 2.0
                    area += fg
                    x += fg * (lat + lat1) / 3.0
                    y += fg * (lng + lng1) / 3.0
                x = x / area
                y = y / area
                xycenter[j,k,0]=x
                xycenter[j,k,1]=y
        return xycenter

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(2, device=self.device)  # box, cls, dfl

        feats,_,_ = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = feats[0].shape  # batch size, number of masks, mask height, mask width

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.nm, self.nc), 1)
        
        new_pred_distri=pred_distri[:,:36,:]

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        new_pred_distri = new_pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)

        anchor_points, stride_tensor,ss = make_anchors_polar(feats, self.stride, 0.5)
        # targets
        try:
            batch_idx = batch['batch_idx'].view(-1, 1)
            import numpy as np


            seg=torch.cat(batch['segments']).contiguous().view(-1,360*2).to(batch_idx.device)

            targets = torch.cat((batch_idx, batch['cls'].contiguous().view(-1, 1), batch['bboxes'],seg), 1)#batch['bboxes']  n,4  
            # print(targets.shape)

            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])


            gt_labels, gt_bboxes,gt_coor = targets.split((1, 4,360*2), 2)  # cls, xyxy
            gt_center=0
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
            # oriimg=batch['img']*255
            # oriimg = oriimg.permute(0,2,3,1).cpu().detach().numpy()
            # cv2.imwrite('/home/data2/mxl/ultralytics/ultralytics-main/output_high_quality.jpg', oriimg[0])


        except RuntimeError as e:

            raise TypeError('ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n'
                            "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                            "i.e. 'yolo train model=yolov8n-seg.pt data=coco128.yaml'.\nVerify your dataset is a "
                            "correctly formatted 'segment' dataset using 'data=coco128-seg.yaml' "
                            'as an example.\nSee https://docs.ultralytics.com/tasks/segment/ for help.') from e
        # pboxes
        new_pred_distri=new_pred_distri*(stride_tensor.detach())
        n_max_boxes = gt_bboxes.size(1)
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx,target_dist,centerness_gt,fg_mask222 = self.assigner(
            pred_scores.detach().sigmoid(), (new_pred_distri.detach()).type(gt_bboxes.dtype),
            (anchor_points * stride_tensor).detach(), gt_labels, gt_bboxes, mask_gt,gt_coor,stride_tensor,ss,gt_center,imgsz)
        


        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # masks loss
            new_pred_distri = new_pred_distri.unsqueeze(1).expand(-1, n_max_boxes,-1, -1)[fg_mask.bool()]
            target_scores = target_scores.unsqueeze(1).expand(-1, n_max_boxes,-1,-1)[fg_mask.bool()]#n,9
            # pred_centerness = pred_centerness.unsqueeze(1).expand(-1, n_max_boxes,-1, -1)[fg_mask.bool()]
            loss[0]=self.polar_loss(new_pred_distri,target_dist,target_scores,target_scores_sum)            

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)






# Criterion class for computing training losses
class oriv8SegmentationLoss(v8DetectionLoss):

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.nm = model.model[-1].nm  # number of masks
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        # print(feats[0].shape)
        # print(pred_masks.shape)

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch['batch_idx'].view(-1, 1)
            targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        except RuntimeError as e:
            raise TypeError('ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n'
                            "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                            "i.e. 'yolo train model=yolov8n-seg.pt data=coco128.yaml'.\nVerify your dataset is a "
                            "correctly formatted 'segment' dataset using 'data=coco128-seg.yaml' "
                            'as an example.\nSee https://docs.ultralytics.com/tasks/segment/ for help.') from e

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # zeromat=torch.zeros_like(pred_scores,dtype=pred_scores.dtype)#(B,numanchor,numclass) 
        # init_w1=torch.ones_like(pred_scores,dtype=pred_scores.dtype)#(B,numanchor,numclass)
        # init_w2=torch.ones_like(pred_scores,dtype=pred_scores.dtype)#(B,numanchor,numclass) 

        # zb = gt_bboxes.sum(2).gt_(0) #fliter zero instance image in batch, zb shape =(b,n) n is fixed instance dim
        # zb = zb.sum(1)==0.0 #shape is (n,)
        # zb=zb.view(-1,1,1).repeat_interleave(pred_scores.shape[2],dim=2)#(B,1,numclass) 
        # zb=zb.repeat_interleave(pred_scores.shape[1],dim=1)#(B,numanchor,numclass) 
        # zb[...,1:]=0
        # init_w1[zb.bool()] =zeromat[zb.bool()]

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        #target_labels.shape is (B,numanchor)  for example : 2,8400
        #pred_scores.shape is (B,numanchor,numclass)  for example : 2,8400,8

        # ind_bb=gt_labels.squeeze(-1).sum(-1)>0.0#find xiangao images.
        # ind_bb=ind_bb.view(-1,1,1).repeat_interleave(pred_scores.shape[2],dim=2)#(B,1,numclass) 
        # ind_bb=ind_bb.repeat_interleave(pred_scores.shape[1],dim=1)#(B,numanchor,numclass) 
        # ind_bb[...,1:]=0
        # init_w2[ind_bb.bool()] = zeromat[ind_bb.bool()]
        # init_w=init_w1*init_w2
        # loss[2] = (init_w*self.bce(pred_scores, target_scores.to(dtype))).sum() / target_scores_sum  # BCE
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # bbox loss
            loss[0], loss[3] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                              target_scores, target_scores_sum, fg_mask)

            # new_target_scores=target_scores.clone()
            # nwe_fg_mask=fg_mask.clone()
            # nwe_target_bboxes=target_bboxes.clone()            
            # target_labels=target_labels.unsqueeze(dim=-1)
            # mask = (target_labels[..., 0] == 0)# other class in addition to tree class

            # nwe_fg_mask=(mask*nwe_fg_mask).bool()
            # nwe_target_bboxes=mask.unsqueeze(dim=-1)*nwe_target_bboxes

            # masks loss
            masks = batch['masks'].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]
            for i in range(batch_size):
                if fg_mask[i].sum():
                # if nwe_fg_mask[i].sum():
                    mask_idx = target_gt_idx[i][fg_mask[i]]
                    if self.overlap:
                        gt_mask = torch.where(masks[[i]] == (mask_idx + 1).view(-1, 1, 1), 1.0, 0.0)
                    else:
                        gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                    xyxyn = target_bboxes[i][fg_mask[i]] / imgsz[[1, 0, 1, 0]]
                    marea = xyxy2xywh(xyxyn)[:, 2:].prod(1)
                    mxyxy = xyxyn * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=self.device)
                    # print(gt_mask.shape)
                    # print(pred_masks.shape)

                    loss[1] += self.single_mask_loss(gt_mask, pred_masks[i][fg_mask[i]], proto[i], mxyxy, marea)  # seg

                # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
                else:
                    loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box / batch_size  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def single_mask_loss(self, gt_mask, pred, proto, xyxy, area):
        """Mask loss for one image."""
        pred_mask = (pred @ proto.view(self.nm, -1)).view(-1, *proto.shape[1:])  # (n, 32) @ (32,80,80) -> (n,80,80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()


# Criterion class for computing training losses
class v8PoseLoss(v8DetectionLoss):

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            for i in range(batch_size):
                if fg_mask[i].sum():
                    idx = target_gt_idx[i][fg_mask[i]]
                    gt_kpt = keypoints[batch_idx.view(-1) == i][idx]  # (n, 51)
                    gt_kpt[..., 0] /= stride_tensor[fg_mask[i]]
                    gt_kpt[..., 1] /= stride_tensor[fg_mask[i]]
                    area = xyxy2xywh(target_bboxes[i][fg_mask[i]])[:, 2:].prod(1, keepdim=True)
                    pred_kpt = pred_kpts[i][fg_mask[i]]
                    kpt_mask = gt_kpt[..., 2] != 0
                    loss[1] += self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss
                    # kpt_score loss
                    if pred_kpt.shape[-1] == 3:
                        loss[2] += self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose / batch_size  # pose gain
        loss[2] *= self.hyp.kobj / batch_size  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def kpts_decode(self, anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y


class v8ClassificationLoss:

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch['cls'], reduction='sum') / 64
        loss_items = loss.detach()
        return loss, loss_items
# class v8ClassificationLoss:
#     def __call__(self, preds, batch):
#         loss_func_sum = nn.CrossEntropyLoss(reduction="sum")
#         loss = loss_func_sum(preds, batch['cls']) / preds.shape[0]
#         loss_items = loss.detach()
#         return loss, loss_items
