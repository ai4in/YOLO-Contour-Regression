# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
from copy import deepcopy
from .checks import check_version
from .metrics import bbox_iou
import numpy as np 
TORCH_1_10 = check_version(torch.__version__, '1.10.0')

def points_in_polygon_align(pts, polygon):
    """
    pts[m, 2]
    polygon[m, n, 2]  [b,maxbox,36*2]
    return [m, ] --- (Tensor): shape(b, n_boxes, m)
    """
    # roll
    b,mb,_=polygon.shape
    na=pts.shape[0]
    polygon=polygon.reshape(b,mb,1,360,2).expand(-1,-1,na,-1,-1)

    contour2 = torch.roll(polygon, -1, 3)
    # print(contour2.shape)

    test_diff = contour2 - polygon#[b,maxbox,na,36,2]

    # [m, n, 2] -> [m, ]
    mask1 = (pts[None,None,:,None] == polygon).all(-1).any(-1)#[b,maxbox,na]
    # print(mask1.shape)
    # print((pts[None,None,:,None] == polygon).shape)

    # [m, n]
    m1 = (polygon[...,1] > pts[None,None,:,None,1]) != (contour2[...,1] > pts[None,None,:,None,1])
    slope = ((pts[None,None,:,None,0]-polygon[...,0])*test_diff[...,1])-(
             test_diff[...,0]*(pts[None,None,:,None,1]-polygon[...,1]))#[b,maxbox,na,36]
    m2 = slope == 0
    mask2 = (m1 & m2).any(-1)#[b,maxbox,na]
    # print(mask2.shape)
    # print(l)

    m3 = (slope < 0) != (contour2[...,1] < polygon[...,1])#[b,maxbox,na,36]
    m4 = m1 & m3#[b,maxbox,na,36]
    count = torch.count_nonzero(m4, dim=-1)
    mask3 = ~(count%2==0)
    mask = mask1 | mask2 | mask3#[b,maxbox,na]
    # print(mask.sum())
    # print(l)
    return mask



def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """select the positive anchor center in gt

    Args:
        xy_centers (Tensor): shape(h*w, 2)
        gt_bboxes (Tensor): shape(b, n_boxes, 4)
    Return:
        (Tensor): shape(b, n_boxes, h*w)
    """
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
    bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
    # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
    return bbox_deltas.amin(3).gt_(eps)




def select_candidates_in_gts_center_box(mask_center,ss,xy_centers, gt_bboxes, eps=1e-9):
    """select the positive anchor center in gt

    Args:
        xy_centers (Tensor): shape(h*w, 2)
        gt_bboxes (Tensor): shape(b, n_boxes, 4)
        mask_center (Tensor): shape(b, n_boxes, 2)
        ss=[A,B,C]
    Return:
        (Tensor): shape(b, n_boxes, h*w)
    """
    na=xy_centers.shape[0]

    gt_xs=xy_centers[...,0]#(b,n_boxes,h*w)
    gt_ys=xy_centers[...,1]
    gt_bboxes = gt_bboxes[:,:,None,:].expand(-1, -1,na ,-1)#(b, n_boxes,na, 4)
    center_x = (gt_bboxes[..., 2] + gt_bboxes[..., 0]) / 2
    center_y = (gt_bboxes[..., 3] + gt_bboxes[..., 1]) / 2
    center_gt = gt_bboxes.new_zeros(gt_bboxes.shape)#(B,n_boxes,na,4)
    radius=1.5
    beg = 0
    for _, level in enumerate(ss):
        num,_=level.shape
        end=beg+num
        stride = level[0]*radius
        xmin = center_x[:,:,beg:end] - stride
        ymin = center_y[:,:,beg:end] - stride
        xmax = center_x[:,:,beg:end] + stride
        ymax = center_y[:,:,beg:end]+ stride
        center_gt[:,:,beg:end,0] = torch.where(xmin > gt_bboxes[:,:,beg:end,0], xmin, gt_bboxes[:,:,beg:end,0])
        center_gt[:,:,beg:end,1] = torch.where(ymin > gt_bboxes[:,:,beg:end,1], ymin, gt_bboxes[:,:,beg:end,1])
        center_gt[:,:,beg:end,2] = torch.where(xmax > gt_bboxes[:,:,beg:end,2], gt_bboxes[:,:,beg:end,2], xmax)
        center_gt[:,:,beg:end,3] = torch.where(ymax > gt_bboxes[:,:,beg:end,3], gt_bboxes[:,:,beg:end,3], ymax)
        beg = end
    
    left = gt_xs - center_gt[..., 0]
    right = center_gt[..., 2] - gt_xs
    top = gt_ys - center_gt[..., 1]
    bottom = center_gt[..., 3] - gt_ys
    center_bbox = torch.stack((left, top, right, bottom), -1)#(b, n_boxes,na,4)
    inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0  # 上下左右都>0 就是在bbox里面  #(b, n_boxes,na)
    return inside_gt_bbox_mask



def select_candidates_in_gts_center(mask_center,ss,xy_centers, gt_bboxes, eps=1e-9):
    """select the positive anchor center in gt

    Args:
        xy_centers (Tensor): shape(h*w, 2)
        gt_bboxes (Tensor): shape(b, n_boxes, 4)
        mask_center (Tensor): shape(b, n_boxes, 2)
        ss=[A,B,C]
    Return:
        (Tensor): shape(b, n_boxes, h*w)
    """
    na=xy_centers.shape[0]
    b,nb,_=mask_center.shape
    xy_centers[None,None,:,:].expand(b, nb,-1 ,-1)
    gt_xs=xy_centers[...,0]#(b,n_boxes,h*w)
    gt_ys=xy_centers[...,0]
    
    mask_center = mask_center[:,:,None,:].expand(-1, -1,na ,-1)#(b, n_boxes,na, 2)
    gt_bboxes=gt_bboxes[:,:,None,:].expand(-1, -1,na ,-1)

    center_x = mask_center[..., 0]#(b, n_boxes,na)
    center_y = mask_center[..., 1]
    # center_y = mask_center[..., 0]
    # center_x = mask_center[..., 1]

    center_gt = gt_bboxes.new_zeros(gt_bboxes.shape)#(B,n_boxes,na,4)
    radius=1
    beg = 0
    for _, level in enumerate(ss):
        num,_=level.shape
        end=beg+num
        stride = level[0]*radius
        # print(stride)
        xmin = center_x[:,:,beg:end] - stride
        ymin = center_y[:,:,beg:end] - stride
        xmax = center_x[:,:,beg:end] + stride
        ymax = center_y[:,:,beg:end]+ stride
        center_gt[:,:,beg:end,0] = torch.where(xmin > gt_bboxes[:,:,beg:end,0], xmin, gt_bboxes[:,:,beg:end,0])
        center_gt[:,:,beg:end,1] = torch.where(ymin > gt_bboxes[:,:,beg:end,1], ymin, gt_bboxes[:,:,beg:end,1])
        center_gt[:,:,beg:end,2] = torch.where(xmax > gt_bboxes[:,:,beg:end,2], gt_bboxes[:,:,beg:end,2], xmax)
        center_gt[:,:,beg:end,3] = torch.where(ymax > gt_bboxes[:,:,beg:end,3], gt_bboxes[:,:,beg:end,3], ymax)
        beg = end
    
    left = gt_xs - center_gt[..., 0]
    right = center_gt[..., 2] - gt_xs
    top = gt_ys - center_gt[..., 1]
    bottom = center_gt[..., 3] - gt_ys
    center_bbox = torch.stack((left, top, right, bottom), -1)#(b, n_boxes,na,4)
    inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0  # 上下左右都>0 就是在bbox里面  #(b, n_boxes,na)

    # print(inside_gt_bbox_mask.sum())
    return inside_gt_bbox_mask

def polarselect_highest_overlaps(mask_pos, overlaps, n_max_boxes,gt_bboxes):
    #polar
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)  0 and 1
        overlaps (Tensor): shape(b, n_max_boxes, h*w)
    Return:
        target_gt_idx (Tensor): shape(b, h*w)
        fg_mask (Tensor): shape(b, h*w)
        mask_pos (Tensor): shape(b, n_max_boxes, h*w) 0 ~n_max_boxes-1
    """
    # (b, n_max_boxes, h*w) -> (b, h*w)
    fg_mask = mask_pos.sum(-2)#(b, h*w)
    wh=gt_bboxes[...,2:]-gt_bboxes[...,:2]
    area=wh[...,0]*wh[...,1]#b,nmax
    area=area.unsqueeze(-1).expand(-1, -1,overlaps.shape[-1])
    area=area.masked_fill(mask_pos==0, 1e8)

    # print('fg_mask',fg_mask)
    if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
        minarea_idx=area.argmin(1)# (b, h*w)
        is_minarea = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
        is_minarea.scatter_(1, minarea_idx.unsqueeze(1), 1)
        mask_pos = torch.where(mask_multi_gts, is_minarea, mask_pos).float()  # (b, n_max_boxes, h*w)

        # max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
        # is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
        # is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
        # mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
        fg_mask = mask_pos.sum(-2)
        # print('mask_multi_gts',mask_multi_gts)
        # print('is_max_overlaps',is_max_overlaps)

    # Find each grid serve which gt(index)
    # print('mask_pos',mask_pos.shape)
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
    # print('target_gt_idx',target_gt_idx.shape)
    # print('max_num_objmask_pos',mask_pos[0,:,:])
    # print('max_num_objmask_pos',target_gt_idx[0,:])

    return target_gt_idx, fg_mask, mask_pos

def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)  0 and 1
        overlaps (Tensor): shape(b, n_max_boxes, h*w)
    Return:
        target_gt_idx (Tensor): shape(b, h*w)
        fg_mask (Tensor): shape(b, h*w)
        mask_pos (Tensor): shape(b, n_max_boxes, h*w) 0 ~n_max_boxes-1
    """
    # (b, n_max_boxes, h*w) -> (b, h*w)
    fg_mask = mask_pos.sum(-2)
    # print('fg_mask',fg_mask)
    if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
        max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

        is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
        is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  #(b, n_max_boxes, h*w)
        fg_mask = mask_pos.sum(-2)
        # print('mask_multi_gts',mask_multi_gts)
        # print('is_max_overlaps',is_max_overlaps)

    # Find each grid serve which gt(index)
    # print('mask_pos',mask_pos.shape)
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
    # print('target_gt_idx',target_gt_idx.shape)
    # print('max_num_objmask_pos',mask_pos[0,:,:])
    # print('max_num_objmask_pos',target_gt_idx[0,:])

    return target_gt_idx, fg_mask, mask_pos



class oriTaskAlignedAssigner(nn.Module):
    #ori
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric,
    which combines both classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment.
        Reference https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)  n_max_boxes的idx
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points,
                                                             mask_gt)

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get in_gts mask, (b, max_num_obj, h*w)."""
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]

        overlaps[mask_gt] = bbox_iou(gt_boxes, pd_boxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.

        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """

        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k:k + 1], ones)
        # count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8, device=topk_idxs.device))
        # filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points.
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.
        """

        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w)
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.num_classes),
                                    dtype=torch.int64,
                                    device=target_labels.device)  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores


class polarTaskAlignedAssigner(nn.Module):
    #polar  基于fcos的标签分配
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric,
    which combines both classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=3.0, eps=1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.regress_ranges=((-1, 64), (64, 128), (128, 1e8))#尺度限制

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt,gt_coor,stride_tensor,ss,gt_center,oriimg,imgsz):
        """
        Compute the task-aligned assignment.
        Reference https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)#有实例的地方
            gt_coor(Tensor):shape(bs, n_max_boxes, 36*2)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)
        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device),0)
        na = pd_bboxes.shape[-2]

        h,w=imgsz
        h=h.item()
        w=w.item()

        device = gt_bboxes.device
        self.all_level_points = [torch.zeros(int(h*w/64),2).to(device),torch.zeros(int(h*w/256),2).to(device),torch.zeros(int(h*w/1024),2).to(device)]#

        expanded_regress_ranges = [
            gt_bboxes.new_tensor(self.regress_ranges[i])[None].expand_as(
                self.all_level_points[i]) for i in range(3)]#6400,2 1600,2 800,2
        
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)#(8400，2）
        concat_regress_ranges = concat_regress_ranges[None,None,:,:].expand(self.bs,
            self.n_max_boxes, -1,-1)#（n,2)->(b,n,gt,2)

        gt_bboxessss = gt_bboxes[:,:,None,:].expand(-1, -1,na ,-1)#(b, n_boxes,na, 4)
        gt_xs=anc_points[...,0]#(h*w)
        gt_ys=anc_points[...,1]
        left = gt_xs[None,None] - gt_bboxessss[..., 0]#
        right = gt_bboxessss[..., 2] - gt_xs[None,None]
        top = gt_ys[None,None] - gt_bboxessss[..., 1]
        bottom = gt_bboxessss[..., 3] - gt_ys[None,None]
        bbbb = torch.stack((left, top, right, bottom), -1)   #feature map上所有点对于gtbox的上下左右距离 [b, n_boxes,na,, 4]
        max_regress_distance = bbbb.max(-1)[0]#[b, n_boxes,na]

        inside_regress_range = (max_regress_distance >= concat_regress_ranges[..., 0]) & (max_regress_distance <= concat_regress_ranges[..., 1])#(bs, n_max_boxes,8400)



        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points,
                                                             mask_gt,deepcopy(gt_coor),ss,gt_center)
        #mask_pos(b, max_num_obj, h*w)

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes,gt_bboxes)
        mask_pos=mask_pos*inside_regress_range
        fg_mask=mask_pos.sum(-2)
        bs,nm,c=gt_coor.shape
        gt_coor=gt_coor.contiguous().view(bs,nm,360,2)
        centerxy=anc_points
        gt_coor = gt_coor.unsqueeze(2).expand(-1, -1, na,-1, -1)[mask_pos.bool()]
        anc_points=anc_points.contiguous().view(1,1,na,1,2).expand(bs, self.n_max_boxes,-1,360, -1)[mask_pos.bool()]
        stride_tensor=stride_tensor.contiguous().view(1,1,na,1).expand(bs, self.n_max_boxes,-1,-1)[mask_pos.bool()]
        theta = torch.arange(0, 360, 360//36).type_as(anc_points)#均匀角度
        angle_per_ct=self.get_angle(anc_points,gt_coor)
        diff_a = (angle_per_ct[:,None]- theta[None, :,None]).abs_()
        diff_a = torch.where(diff_a.gt(180.), 360 - diff_a, diff_a)
        # # 选角度差异最小的topk位置，选择其中距离最长的为 期望角度 的距离
        k=4
        val_a, idx_a = diff_a.topk(k , 2, largest=False)
        # unique1 = val_a.lt(3)#角度小于3的为T
        unique1 = val_a.min(2)[0].gt(3)[...,None].expand(-1,-1,k)
        # print(val_a.min(2)[0].gt(2).shape)
        # print(unique1.shape)

        dist = torch.norm(gt_coor - anc_points, 2, 2)
        dist = dist[:, None].expand(-1, 36, -1)
        dist_a = torch.gather(dist, 2, idx_a)
        # print(dist_a.shape)


        gt_dist=torch.where(unique1,torch.full_like(dist_a, 1e-6),dist_a)#角度差异大于3的赋值为1e-6
        gt_dist, dist_ids = gt_dist.max(2)
        gt_dist=gt_dist.clamp(min=1e-6)
        centerness_gt=self.polar_centerness_target(gt_dist)

        # import cv2
        # # print(gt_dist.shape)
        # # print(torch.nonzero(mask_pos))
        # centeridx=torch.nonzero(mask_pos)
        # orimg=oriimg[0].astype(np.uint8).copy()
        # print('centeridx.shape',centeridx.shape)
        # wqq=pd_bboxes[fg_mask.bool()]
        # for i in range(centeridx.shape[0]):
        #     batch_id,_,grid_id=centeridx[i]
        #     if batch_id==0:
        #         print('centerxy',centerxy)
        #         x,y=centerxy[grid_id]
        #         with open('/home/data2/mxl/ultralytics/ultralytics-main/tensor_data.txt', 'a') as file:
        #             # 写入 tensor B 的数据
        #             file.write(str(wqq[i]) +"\n")
        #             file.write(str(gt_dist[i]) + "\n")
        #         centernessd = (gt_dist[i].min(dim=-1)[0] / gt_dist[i].max(dim=-1)[0])
        #         c_x, c_y = int(x), int(y)
        #         angles = torch.arange(0, 360, 360//36).type_as(dist_a)/180.*np.pi #36
        #         sin = torch.sin(angles)
        #         cos = torch.cos(angles)
        #         segx = gt_dist[i] * cos + c_x
        #         segy = gt_dist[i] * sin + c_y #B,an,36
        #         seg=torch.cat((segx.reshape(36,1),segy.reshape(36,1)),-1)
        #         contour = seg.detach().cpu().numpy().astype(np.int32)
        #         img = np.zeros((640, 640), dtype=np.uint8)
        #         cv2.fillPoly(img, [contour], color=255)
        #         ret, thresh = cv2.threshold(img, 127, 255, 0)

        #         img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        #         contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #         cnt = contours[0]
        #         if centernessd>0.2:
        #             # print(gt_dist[i])
        #             # print(centernessd)
        #             # print(int(x), int(y))
        #             cv2.drawContours(orimg, [cnt], 0, (0, 255, 0), 1)
        #             cv2.circle(orimg,(int(x), int(y)), 1,(255, 0, 0), -1)
        #             cv2.imwrite('/home/data2/mxl/ultralytics/ultralytics-main/bb/ba/sss{}.jpg'.format(i), orimg)
        # # cv2.imwrite('/home/data2/mxl/ultralytics/ultralytics-main/ad.jpg'.format(i), orimg)
        # print(l)





        # centerxy=anc_points
        # gt_coor = gt_coor.unsqueeze(2).expand(-1, -1, na,-1, -1)
        # anc_points=anc_points.contiguous().view(1,1,na,1,2)
        # angle_per_ct=self.get_angle(anc_points,gt_coor)
        # print(angle_per_ct.shape)
        # print(gt_coor[mask_pos.bool()].shape)
        # print(gt_coor.shape)
        # theta = torch.arange(0, 360, 360//36).type_as(anc_points)#均匀角度
        # diff_a = (angle_per_ct[:,:,:,None]- theta[None,None,None, :,None]).abs_()
        # diff_a = torch.where(diff_a.gt(180.), 360 - diff_a, diff_a)
        # print(diff_a.shape)
        # k=4
        # val_a, idx_a = diff_a.topk(k , -1, largest=False)
        # unique1 = val_a.lt(3)#角度小于3的为T
        # dist = torch.norm(gt_coor - anc_points, 2, -1)
        # print(dist.shape)

        # dist = dist[:,:,:,None,:].expand(-1, -1,-1,36, -1)
        # dist_a = torch.gather(dist, 4, idx_a)
        # print('idx_a:',idx_a.shape)
        # print('dist:',dist.shape)
        # print('dist_a:',dist_a.shape)

        # gt_dist=torch.where(unique1,dist_a,torch.full_like(dist_a, 1e-6))#角度差异大于3的赋值为1e-6
        # print(gt_dist.shape)
        # gt_dist, dist_ids = gt_dist.max(-1)
        # gt_dist=gt_dist.clamp(min=1e-6)
        # # gt_dist=gt_dist/stride_tensor
        # # print(anc_points.shape)
        # import cv2
        # print(gt_dist.shape)
        
        # print(gt_dist[0,0,...])
        # print(torch.nonzero(mask_pos[0,...]))
        # centeridx=torch.nonzero(mask_pos[0,...])
        # orimg=oriimg[0].astype(np.uint8).copy()
        # print('centeridx.shape',centeridx.shape)
        # for i in range(centeridx.shape[0]):
        #     batchid=centeridx[i][0]
        #     id=centeridx[i][1]
        #     x,y=centerxy[id]
        #     print(i,id)
        #     print(gt_dist[0,batchid,id])
        #     centernessd = (gt_dist[0,batchid,id].min(dim=-1)[0] / gt_dist[0,batchid,id].max(dim=-1)[0])
        #     print(centernessd)
        #     print(int(x), int(y))
        #     c_x, c_y = int(x), int(y)
        #     angles = torch.arange(0, 360, 360//36).type_as(dist_a)/180.*np.pi #36
        #     sin = torch.sin(angles)
        #     cos = torch.cos(angles)
        #     segx = gt_dist[0,batchid,id] * cos + c_x
        #     segy = gt_dist[0,batchid,id] * sin + c_y #B,an,36
        #     seg=torch.cat((segx.reshape(36,1),segy.reshape(36,1)),-1)
        #     contour = seg.detach().cpu().numpy().astype(np.int32)
        #     img = np.zeros((320, 320), dtype=np.uint8)
        #     cv2.fillPoly(img, [contour], color=255)
        #     ret, thresh = cv2.threshold(img, 127, 255, 0)

        #     img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        #     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #     cnt = contours[0]
        #     if centernessd>0.2:
        #         cv2.drawContours(orimg, [cnt], 0, (0, 255, 0), 1)
        #         cv2.circle(orimg,(int(x), int(y)), 3,(255, 0, 0), -1)
        #         cv2.imwrite('/home/data2/mxl/ultralytics/ultralytics-main/bb/ba/sss{}.jpg'.format(i), orimg)

        # cv2.imwrite('/home/data2/mxl/ultralytics/ultralytics-main/ad.jpg'.format(i), orimg)

        # for i in range(anc_points.shape[0]):
        #     # print(i)
        #     c_x, c_y = anc_points[i,0,0],anc_points[i,0,1]
        #     angles = torch.arange(0, 360, 360//36).type_as(dist_a)/180.*np.pi #36
        #     sin = torch.sin(angles)
        #     cos = torch.cos(angles)
        #     ttff=gt_dist[i].gt(0)#36
        #     segx = gt_dist[i] * cos + c_x
        #     segy = gt_dist[i] * sin + c_y #B,an,36
        #     seg=torch.cat((segx.reshape(36,1),segy.reshape(36,1)),-1)
        #     seg=seg[ttff]
        #     print(seg.shape)

        #     contour = seg.detach().cpu().numpy().astype(np.int32)
        #     img = np.zeros((640, 640), dtype=np.uint8)
        #     cv2.fillPoly(img, [contour], color=255)
        #     img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        #     j=0
        #     for pp in contour:
        #         # print('unique1')
        #         # print(j)
        #         if j <3:
        #             img = cv2.line(img, (contour[j][0], contour[j][1]), (contour[j+1][0], contour[j+1][1]), (0 ,0, 255), 1)
        #             cv2.circle(img,(int(contour[j][0]), int(contour[j][1])), 3,
        #                            (255, 0, 0), -1)  # 在交点处绘制圆点

        #             j=j+1
        #         else:
        #             cv2.circle(img,(int(contour[j][0]), int(contour[j][1])), 3,
        #                            (255, 0, 0), -1)  # 在交点处绘制圆点
        #             j=j+1

        #         # if j ==35:
        #         #     img = cv2.line(img, (contour[j][0], contour[j][1]), (contour[0][0], contour[0][1]), (0 ,0, 255), 1)
        #     cv2.circle(img,(int(c_x), int( c_y)), 3,
        #                            (0, 0, 255), -1)  # 在交点处绘制圆点
        #     cv2.imwrite('/home/data2/mxl/ultralytics/ultralytics-main/bb/ba/sss{}.jpg'.format(i), img)
        # print(l)

        # Assigned target
    
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)
        # print(target_labels)
        # print(target_scores[0,0:100].argmax(-1))
        # print(l)

        # Normalize
        # align_metric *= mask_pos
        # pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)  # b, max_num_obj
        # pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)  # b, max_num_obj

        # pos_overlaps = (overlaps * mask_pos).amax(-2).unsqueeze(-1)  # b, max_num_obj

        # hwIOUS=(overlaps * mask_pos).amax(axis=-2).unsqueeze(-1)
        # print(hwIOUS.shape)
        # target_scores = target_scores *hwIOUS


        # norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        # target_scores = target_scores * norm_align_metric
        # print(centerness_gt.shape)
        # print(target_scores[fg_mask.bool()].shape)
        # print(l)
        # target_scores[fg_mask.bool()] =target_scores[fg_mask.bool()]*centerness_gt.unsqueeze(-1)
        # print(centerness_gt.shape)


        # import cv2
        # print(gt_dist.shape)
        # print(torch.nonzero(mask_pos))
        # centeridx=torch.nonzero(mask_pos)
        # print('centeridx.shape',centeridx.shape)
        # for i in range(centeridx.shape[0]):
        #     batch_id,instance_id,grid_id=centeridx[i]
        #     if batch_id==1:
        #         orimg=oriimg[batch_id].astype(np.uint8).copy()
        #         print('target_scores:',target_scores[batch_id][grid_id])
        #         score=target_scores[batch_id][grid_id].argmax()
        #         print('score:',score)
        #         x,y=centerxy[grid_id]           
        #         centernessd = torch.sqrt((gt_dist[i].min(dim=-1)[0] / gt_dist[i].max(dim=-1)[0]))
        #         c_x, c_y = int(x), int(y)
        #         angles = torch.arange(0, 360, 360//36).type_as(dist_a)/180.*np.pi #36
        #         sin = torch.sin(angles)
        #         cos = torch.cos(angles)
        #         segx = gt_dist[i] * cos + c_x
        #         segy = gt_dist[i] * sin + c_y #B,an,36
        #         seg=torch.cat((segx.reshape(36,1),segy.reshape(36,1)),-1)
        #         contour = seg.detach().cpu().numpy().astype(np.int32)
        #         img = np.zeros((640, 640), dtype=np.uint8)
        #         cv2.fillPoly(img, [contour], color=255)
        #         ret, thresh = cv2.threshold(img, 127, 255, 0)

        #         img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        #         contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #         cnt = contours[0]
        #         # if centernessd>0.2:
        #         print(gt_dist[i])
        #         print(centernessd)
        #         # print(int(x), int(y))
        #         if centernessd>0.2:
        #             cv2.drawContours(orimg, [cnt], 0, (0, 255, 0), 1)
        #             cv2.circle(orimg,(int(x), int(y)), 1,(255, 0, 0), -1)
        #             cv2.putText(orimg, '{}'.format(score), (c_x, c_y), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 128), 2, cv2.LINE_AA)
        #             cv2.imwrite('/home/data2/mxl/ultralytics/ultralytics-main/bb/ba/sss{}.jpg'.format(i), orimg)
        # cv2.imwrite('/home/data2/mxl/ultralytics/ultralytics-main/ad.jpg'.format(i), orimg)

        return target_labels, target_bboxes, target_scores, mask_pos.bool(), target_gt_idx,gt_dist,centerness_gt,fg_mask.bool()

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt,gt_coor,ss,gt_center):
        """Get in_gts mask, (b, max_num_obj, h*w)."""
        mask_in_gts = select_candidates_in_gts_center_box(gt_center,ss,anc_points, gt_bboxes)
        # mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)#mask_in_gts (b, max_num_obj, h*w)
        # wh=gt_bboxes[...,2:]-gt_bboxes[...,:2]
        # area=wh[...,0]*wh[...,1]

        # mask_in_gts=points_in_polygon_align(anc_points,gt_coor)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        # align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        align_metric, overlaps = self.get_box_metrics_polar(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt,gt_coor,anc_points)

        # Get topk_metric mask, (b, max_num_obj, h*w)
        # mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        # mask_pos = mask_topk * mask_in_gts * mask_gt
        mask_pos = mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps


    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = bbox_iou(gt_boxes, pd_boxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def polar_centerness_target(self, pos_mask_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        centerness_targets = (pos_mask_targets.min(dim=-1)[0] / pos_mask_targets.max(dim=-1)[0])
        # a=pos_mask_targets.min(dim=-1)[0]
        # print(a[a>1.0000e-06])
        # print(pos_mask_targets.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
    
    def polar_centerness_targetv2(self, pos_mask_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        centerness_targets = (pos_mask_targets.min(dim=-1)[0] / pos_mask_targets.max(dim=-1)[0])
        # a=pos_mask_targets.min(dim=-1)[0]
        # print(a[a>1.0000e-06])
        # print(pos_mask_targets.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)


    def get_box_metrics_polar(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt,gt_ray,anc_points):
        #gt_ray (bs, n_max_boxes, 360,2)
        #anc_points (Tensor): shape(num_total_anchors, 2)
        bs,nm,c=gt_ray.shape
        gt_ray=gt_ray.contiguous().view(bs,nm,360,2)

        """Compute alignment metric given predicted and ground truth bounding boxes."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # print(pd_scores[ind[0], :, ind[1]].shape)
        # print(ind[0])
        # print(ind[1])
        # print(pd_scores[0,0:20,:])
        # print(pd_scores[ind[0], :, ind[1]][0,:,0:20])

        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        # print('gt_ray',gt_ray.unsqueeze(2).expand(-1, -1, na,-1, -1).shape)
        # print('pd_boxes',pd_boxes.shape)
        gt_ray = gt_ray.unsqueeze(2).expand(-1, -1, na,-1, -1)[mask_gt]
        # print('gt_ray',gt_ray.shape)
        anc_points=anc_points.contiguous().view(1,1,na,1,2).expand(bs, self.n_max_boxes,-1,360, -1)[mask_gt]
        # print('anc_points',anc_points.shape)
        # del anc_points

        angle_per_ct=self.get_angle(anc_points,gt_ray)#(na,36)

        theta = torch.arange(0, 360, 360//36).type_as(anc_points)#均匀角度
        diff_a = (angle_per_ct[:,None]- theta[None, :,None]).abs_()#(na,1,36)-(na,36,1)
        diff_a = torch.where(diff_a.gt(180.), 360 - diff_a, diff_a)

        # 选角度差异最小的topk位置，选择其中距离最长的为 期望角度 的距离
        k=4
        val_a, idx_a = diff_a.topk(k , 2, largest=False)
        unique1 = val_a.min(2)[0].gt(3)[...,None].expand(-1,-1,k)
        del diff_a
        dist = torch.norm(gt_ray - anc_points, 2, 2)#(na,36)
        dist = dist[:, None].expand(-1, 36, -1)#(na,36,36)
        dist_a = torch.gather(dist, 2, idx_a)
        dist_a=torch.where(unique1,torch.full_like(dist_a, 1e-6),dist_a)#角度差异大于3的赋值为1e-6
        dist_max, dist_ids = dist_a.max(2)
        dist_max=dist_max.clamp(min=1e-6)



        # print('val_a.min(2)[0].gt(2)[...,None].shape',val_a.min(2)[0].gt(2)[...,None].shape)
        # print('val_a.min(2)[0].gt(2)[...,None].shape',val_a.min(2)[0].gt(2)[...,None])

        # 这个版本有可能pts在box里面但是不再conts里面
        # 所以对于点在conts外面的，把那些最小角度大于2度的设为0,因为射线没有负的
        # val_bool = ((~pts_in_plg[...,None,None]) & val_a.min(2)[0].gt(2)[...,None]
        #             ).expand(-1,-1,k)
        # dist_a = torch.where(val_bool,
        #                         torch.full_like(dist_a, 1e-6),
        #                         dist_a)
        # print(dist_a)
        # print(dist_max.shape)

        # print(pd_boxes.shape)
        centerness=self.polar_centerness_target(dist_max)
        # centerness_pred=self.polar_centerness_target(pd_boxes)

        # print(centerness.shape)
        # print(dist_max[:5])

        # print(l)
        # print(dist_max[0])
        # import cv2
        # for i in range(anc_points.shape[0]):
        #     c_x, c_y = anc_points[i,0,0],anc_points[i,0,1]
        #     angles = torch.arange(0, 360, 360//36).type_as(dist_a)/180.*np.pi #36
        #     sin = torch.sin(angles)
        #     cos = torch.cos(angles)
        #     ttff=dist_max[i].gt(1)#36
        #     segx = dist_max[i] * cos + c_x
        #     segy = dist_max[i] * sin + c_y #B,an,36
        #     seg=torch.cat((segx.reshape(36,1),segy.reshape(36,1)),-1)
        #     seg=seg[ttff]
        #     # print(seg.shape)

        #     contour = seg.detach().cpu().numpy().astype(np.int32)
        #     img = np.zeros((128, 128), dtype=np.uint8)
        #     cv2.fillPoly(img, [contour], color=255)
        #     img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)


        #     j=0
        #     for pp in contour:
        #         print('unique1')
        #         print(j)
        #         if j <3:
        #             img = cv2.line(img, (contour[j][0], contour[j][1]), (contour[j+1][0], contour[j+1][1]), (0 ,0, 255), 1)
        #             cv2.circle(img,(int(contour[j][0]), int(contour[j][1])), 3,
        #                            (255, 0, 0), -1)  # 在交点处绘制圆点

        #             j=j+1
        #         else:
        #             cv2.circle(img,(int(contour[j][0]), int(contour[j][1])), 3,
        #                            (255, 0, 0), -1)  # 在交点处绘制圆点
        #             j=j+1

        #         # if j ==35:
        #         #     img = cv2.line(img, (contour[j][0], contour[j][1]), (contour[0][0], contour[0][1]), (0 ,0, 255), 1)
        #     cv2.circle(img,(int(c_x), int( c_y)), 3,
        #                            (0, 0, 255), -1)  # 在交点处绘制圆点
        #     cv2.imwrite('/home/data2/mxl/ultralytics/ultralytics-main/bb/bb1/sss{}.jpg'.format(i), img)
        # print(l)

        # print(pd_boxes.shape)
        overlaps[mask_gt] = MaskIOU(dist_max, pd_boxes)*centerness# b, max_num_obj, h*w
        # print(overlaps[overlaps>0])

        # print(centerness[centerness>0.1].shape)
        # print("centerness[centerness>0.1")
        # print(centerness_pred[centerness_pred>0.1])
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        # align_metric[mask_gt]=centerness_pred*align_metric[mask_gt]

        # print(overlaps[overlaps>0.1])

        # bbox_scores[mask_gt]
        # print(l)

        # print(overlaps[overlaps>0])
        # print(l)
        return align_metric, overlaps

    def get_angle(self,ct_pts, conts):
        """
        ct_pts[B,m,n, 2]
        conts[B,m, n, 2]
        计算m个中心点与n个定点围成的角度
        # 输出：[m, n]
        """
        v = conts - ct_pts#[B,m,n, 2]
        # 与x轴正方向的夹角，顺时针为正,atan2输出为(-180,180）
        angle = torch.atan2(v[..., 1], v[..., 0])##[B,m,n, 1]
        
        angle = angle * 180. / np.pi#转成度数制
        included_angle = torch.where(
            angle<0, angle+360, angle)
        
        return included_angle


    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.

        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """

        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k:k + 1], ones)
        # count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8, device=topk_idxs.device))
        # filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.
            gt_coor (Tensor): tensor of shape (b, max_num_obj,36*2) 

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points.
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.
        """

        # Assigned target labels, (b, 1)
        # print('max_num_obj',gt_labels.shape)
        # b,_,_=gt_bboxes.shape
        # na=anc_points.shape[0]
        # anc_points=anc_points.contiguous().view(1,1,na*2).expand(b, self.n_max_boxes,-1)#b,max_num_obj,na*2

        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]#B,1
        # print('batch_ind',batch_ind)

        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        # print('gt_labels',gt_labels)
        # print('gt_labels',gt_labels.long().flatten())
        # print('gt_labels',target_gt_idx[1])
        # print(target_gt_idx)
        # print(target_gt_idx.shape)

        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)  从gt_labels中按照target_gt_idx找值
        # print(target_labels[1])
        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w)
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]
        # gt_coor = gt_coor.view(-1, 36*2)[target_gt_idx]
        # anc_points = anc_points.view(-1, na*2)[target_gt_idx]

        # print('target_gt_idx',target_gt_idx.shape)
        # print('target_gt_idx',target_gt_idx)
        # print('gt_bboxes',gt_bboxes.shape)
        # print('gt_bboxes',gt_bboxes)
        # print('target_bboxes',target_bboxes.shape)
        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        # print('target_labels.shape',target_labels.shape)

        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.num_classes),
                                    dtype=torch.int64,
                                    device=target_labels.device)  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)#将target_lable 变为one-hot
        # print('target_scores.shape',target_scores.shape)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)#筛选
        # print('target_scores.shape',target_scores.shape)
        # print('target_scores.shape',target_scores)
        # print(l)
        return target_labels, target_bboxes, target_scores

class TaskAlignedAssigner(nn.Module):
    #polarpaper（v8标签分配）
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric,
    which combines both classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=3.0, eps=1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt,gt_coor,stride_tensor,ss,gt_center,imgsz):
        """
        Compute the task-aligned assignment.
        Reference https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)#有实例的地方
            gt_coor(Tensor):shape(bs, n_max_boxes, 36*2)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)
        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device),0)
        na = pd_bboxes.shape[-2]


        device = gt_bboxes.device
        
        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points,
                                                             mask_gt,deepcopy(gt_coor),ss,gt_center)
        #mask_pos(b, max_num_obj, h*w)

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)
        bs,nm,c=gt_coor.shape
        gt_coor=gt_coor.contiguous().view(bs,nm,360,2)
        centerxy=anc_points
        gt_coor = gt_coor.unsqueeze(2).expand(-1, -1, na,-1, -1)[mask_pos.bool()]
        anc_points=anc_points.contiguous().view(1,1,na,1,2).expand(bs, self.n_max_boxes,-1,360, -1)[mask_pos.bool()]
        stride_tensor=stride_tensor.contiguous().view(1,1,na,1).expand(bs, self.n_max_boxes,-1,-1)[mask_pos.bool()]
        theta = torch.arange(0, 360, 360//36).type_as(anc_points)#均匀角度
        angle_per_ct=self.get_angle(anc_points,gt_coor)
        diff_a = (angle_per_ct[:,None]- theta[None, :,None]).abs_()
        diff_a = torch.where(diff_a.gt(180.), 360 - diff_a, diff_a)
        # # 选角度差异最小的topk位置，选择其中距离最长的为 期望角度 的距离
        k=4
        val_a, idx_a = diff_a.topk(k , 2, largest=False)
        unique1 = val_a.min(2)[0].gt(3)[...,None].expand(-1,-1,k)
        dist = torch.norm(gt_coor - anc_points, 2, 2)
        dist = dist[:, None].expand(-1, 36, -1)
        dist_a = torch.gather(dist, 2, idx_a)
        gt_dist=torch.where(unique1,torch.full_like(dist_a, 1e-6),dist_a)#角度差异大于3的赋值为1e-6
        gt_dist, dist_ids = gt_dist.max(2)
        gt_dist=gt_dist.clamp(min=1e-6)
        # gt_dist=gt_dist/stride_tensor
        centerness_gt=self.polar_centerness_target(gt_dist)
    
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, mask_pos.bool(), target_gt_idx,gt_dist,centerness_gt,fg_mask.bool()

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt,gt_coor,ss,gt_center):
        """Get in_gts mask, (b, max_num_obj, h*w)."""
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)#mask_in_gts (b, max_num_obj, h*w)

        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics_polar(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt,gt_coor,anc_points)

        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt
        # mask_pos = mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def polar_centerness_target(self, pos_mask_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        centerness_targets = (pos_mask_targets.min(dim=-1)[0] / pos_mask_targets.max(dim=-1)[0])
        # a=pos_mask_targets.min(dim=-1)[0]
        # print(a[a>1.0000e-06])
        # print(pos_mask_targets.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
    
    def polar_centerness_targetv2(self, pos_mask_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        centerness_targets = (pos_mask_targets.min(dim=-1)[0] / pos_mask_targets.max(dim=-1)[0])
        # a=pos_mask_targets.min(dim=-1)[0]
        # print(a[a>1.0000e-06])
        # print(pos_mask_targets.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)


    def get_box_metrics_polar(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt,gt_ray,anc_points):
        #gt_ray (bs, n_max_boxes, 360,2)
        #anc_points (Tensor): shape(num_total_anchors, 2)
        bs,nm,c=gt_ray.shape
        gt_ray=gt_ray.contiguous().view(bs,nm,360,2)

        """Compute alignment metric given predicted and ground truth bounding boxes."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj

        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_ray = gt_ray.unsqueeze(2).expand(-1, -1, na,-1, -1)[mask_gt]
        anc_points=anc_points.contiguous().view(1,1,na,1,2).expand(bs, self.n_max_boxes,-1,360, -1)[mask_gt]

        angle_per_ct=self.get_angle(anc_points,gt_ray)#(na,36)

        theta = torch.arange(0, 360, 360//36).type_as(anc_points)#均匀角度
        diff_a = (angle_per_ct[:,None]- theta[None, :,None]).abs_()#(na,1,36)-(na,36,1)
        diff_a = torch.where(diff_a.gt(180.), 360 - diff_a, diff_a)

        # 选角度差异最小的topk位置，选择其中距离最长的为 期望角度 的距离
        k=4
        val_a, idx_a = diff_a.topk(k , 2, largest=False)
        unique1 = val_a.min(2)[0].gt(3)[...,None].expand(-1,-1,k)
        del diff_a
        dist = torch.norm(gt_ray - anc_points, 2, 2)#(na,36)
        dist = dist[:, None].expand(-1, 36, -1)#(na,36,36)
        dist_a = torch.gather(dist, 2, idx_a)
        dist_a=torch.where(unique1,torch.full_like(dist_a, 1e-6),dist_a)#角度差异大于3的赋值为1e-6
        dist_max, dist_ids = dist_a.max(2)
        dist_max=dist_max.clamp(min=1e-6)
        # centerness=self.polar_centerness_target(dist_max)
        # overlaps[mask_gt] = MaskIOU(dist_max, pd_boxes)*centerness# b, max_num_obj, h*w
        overlaps[mask_gt] = MaskIOU(dist_max, pd_boxes)# b, max_num_obj, h*w
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        # align_metric[mask_gt]=centerness_pred*align_metric[mask_gt]

        return align_metric, overlaps

    def get_angle(self,ct_pts, conts):
        """
        ct_pts[B,m,n, 2]
        conts[B,m, n, 2]
        计算m个中心点与n个定点围成的角度
        # 输出：[m, n]
        """
        v = conts - ct_pts#[B,m,n, 2]
        # 与x轴正方向的夹角，顺时针为正,atan2输出为(-180,180）
        angle = torch.atan2(v[..., 1], v[..., 0])##[B,m,n, 1]
        
        angle = angle * 180. / np.pi#转成度数制
        included_angle = torch.where(
            angle<0, angle+360, angle)
        
        return included_angle


    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.

        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """

        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k:k + 1], ones)
        # count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8, device=topk_idxs.device))
        # filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.
            gt_coor (Tensor): tensor of shape (b, max_num_obj,36*2) 

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points.
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.
        """

        # Assigned target labels, (b, 1)
        # print('max_num_obj',gt_labels.shape)
        # b,_,_=gt_bboxes.shape
        # na=anc_points.shape[0]
        # anc_points=anc_points.contiguous().view(1,1,na*2).expand(b, self.n_max_boxes,-1)#b,max_num_obj,na*2

        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]#B,1

        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)

        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)  从gt_labels中按照target_gt_idx找值
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]
        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()

        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.num_classes),
                                    dtype=torch.int64,
                                    device=target_labels.device)  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)#将target_lable 变为one-hot

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)#筛选
        return target_labels, target_bboxes, target_scores


def make_anchors_polar(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    # print(strides)
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        # print(h,w)
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor),stride_tensor

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    # print(strides)
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        # print(h,w)
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)




def MaskIOU(target,pred):
    '''
        :param pred:  shape (N,36), N is nr_box
        :param target: shape (N,36)
        :return: loss
        '''
    N0,ray_size=target.shape


    total = torch.stack([pred,target], -1)#N0,ray_size,2
    l_max = total.max(dim=-1)[0]#N0,ray_size
    l_min = total.min(dim=-1)[0].clamp(min=1e-6)

    # loss = (l_max.sum(dim=-1) / l_min.sum(dim=-1))#b,N,H*W
    loss = (l_min.sum(dim=-1) / l_max.sum(dim=-1))#b,N,H*W

    # print('loss',loss)
    # print('loss',loss.shape)

    return loss


def MaskIOU_centerness(target,pred):
    '''
        :param pred:  shape (N,36), N is nr_box
        :param target: shape (N,36)
        :return: loss
        '''
    N0,ray_size=target.shape

    premin=pred.min(dim=-1)[0]
    premax=pred.max(dim=-1)[0]
    tmin=target.min(dim=-1)[0]
    tmax=target.max(dim=-1)[0]
    pred=torch.stack([premin,premax], -1)
    target=torch.stack([tmin,tmax], -1)


    total = torch.stack([pred,target], -1)#N0,ray_size,2
    l_max = total.max(dim=-1)[0]#N0,ray_size
    l_min = total.min(dim=-1)[0].clamp(min=1e-6)

    # loss = (l_max.sum(dim=-1) / l_min.sum(dim=-1))#b,N,H*W
    loss = (l_min.sum(dim=-1) / l_max.sum(dim=-1))#b,N,H*W

    # print('loss',loss)
    # print('loss',loss.shape)

    return loss

