# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Model head modules
"""

import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_
import torch.nn.functional as F
import torch.nn.init as init

from ultralytics.utils.tal import dist2bbox, make_anchors
from ultralytics.utils.checks import check_version
import numpy as np 
from .block import DFL, Proto
from .conv import Conv,DWConv,RepConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init_
TORCH_1_10 = check_version(torch.__version__, '1.10.0')

__all__ = 'Detect', 'Segment', 'Pose', 'Classify', 'RTDETRDecoder'

class oriDetect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export:
            # box_value=self.dfl(box)

            cls=F.interpolate(cls.unsqueeze(1),scale_factor=1).squeeze(1)
            # mc=F.interpolate(mc.unsqueeze(1),scale_factor=1).squeeze(1)
            box_value=self.dfl(box)
            # return (box_value, cls,mc)#tree
            return (box_value, cls)

        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)
        # return (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

class treeandcarDetect(nn.Module):
    #treeandcar
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nm=8#tree
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4# number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        # self.cv2 = nn.ModuleList(
        #     nn.Sequential(RepConv(x, c2//4, 3), RepConv(c2//4, c2//4, 3), nn.Conv2d(c2//4, 4 * self.reg_max+self.nc, 1)) for x in ch)
        # self.cv2 = nn.ModuleList(#
        #     nn.Sequential(RepConv(x, c2//4, 3), nn.Conv2d(c2//4, 4 * self.reg_max+self.nc+self.nm, 1)) for x in ch)#tree
        self.cv2 = nn.ModuleList(
            nn.Sequential(RepConv(x, c2//4, 3), nn.Conv2d(c2//4, 4 * self.reg_max+self.nc, 1)) for x in ch)

        # self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = self.cv2[i](x[i])
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        # x_cat = torch.cat([xi.view(shape[0], self.no+self.nm, -1) for xi in x], 2)#tree
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)#b,c,8400

        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:self.reg_max * 4+self.nc]
            mc= x_cat[:, self.reg_max * 4+self.nc:]#tree
        else:
            # box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:self.reg_max * 4+self.nc]
            mc= x_cat[:, self.reg_max * 4+self.nc:]#tree

        if self.export:
            # box_value=self.dfl(box)

            cls=F.interpolate(cls.unsqueeze(1),scale_factor=1).squeeze(1)
            # mc=F.interpolate(mc.unsqueeze(1),scale_factor=1).squeeze(1)
            box_value=self.dfl(box)
            # return (box_value, cls,mc)#tree
            return (box_value, cls)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides


        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)

        return  (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, s in zip(m.cv2, m.stride):  # from
            a[-1].bias.data[:4 * self.reg_max] = 1.0  # box
            a[-1].bias.data[4 * self.reg_max:4 * self.reg_max+m.nc] = math.log(5 / m.nc / (128 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)





class Detect(nn.Module):
    #NOFPN-1head
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nm=8#tree
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4# number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        # self.cv2 = nn.ModuleList(
        #     nn.Sequential(RepConv(x, c2//4, 3), RepConv(c2//4, c2//4, 3), nn.Conv2d(c2//4, 4 * self.reg_max+self.nc, 1)) for x in ch)
        # self.cv2 = nn.ModuleList(#
        #     nn.Sequential(RepConv(x, c2//4, 3), nn.Conv2d(c2//4, 4 * self.reg_max+self.nc+self.nm, 1)) for x in ch)#tree
        self.cv2 = nn.ModuleList(
            nn.Sequential(RepConv(x, c2//4, 3), nn.Conv2d(c2//4, 4 * self.reg_max+self.nc, 1)) for x in ch)

        self.cv3 =  nn.ModuleList(
            nn.Sequential(RepConv(x, c2//4, 3), nn.Conv2d(c2//4, self.reg_max*2, 1)) for x in ch)

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        # x_cat = torch.cat([xi.view(shape[0], self.no+self.nm, -1) for xi in x], 2)#tree
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)#b,c,8400

        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:self.reg_max * 4+self.nc]
            mc= x_cat[:, self.reg_max * 4+self.nc:]#tree
        else:
            # box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:self.reg_max * 4+self.nc]
            mc= x_cat[:, self.reg_max * 4+self.nc:]#tree

        if self.export:
            # box_value=self.dfl(box)

            cls=F.interpolate(cls.unsqueeze(1),scale_factor=1).squeeze(1)
            # mc=F.interpolate(mc.unsqueeze(1),scale_factor=1).squeeze(1)
            box_value=self.dfl(box)
            # return (box_value, cls,mc)#tree
            return (box_value, cls)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides


        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)

        return  (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, s in zip(m.cv2, m.stride):  # from
            a[-1].bias.data[:4 * self.reg_max] = 1.0  # box
            a[-1].bias.data[4 * self.reg_max:4 * self.reg_max+m.nc] = math.log(5 / m.nc / (128 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)










class polarDetect(nn.Module):
    # polar fcos标签分配
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):  # detection layer

        super().__init__()
        self.nm=36+1
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.nm  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        # self.cv2 = nn.ModuleList(
        #     # nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, self.nc+self.nm, 1)) for x in ch)
        #     nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, self.nc+self.nm, 1)) for x in ch)

        self.cv2 = nn.ModuleList(
            # nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, self.nc+self.nm, 1)) for x in ch)
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, self.nm, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            # x[i] = self.cv2[i](x[i])
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.nm]
            cls = x_cat[:, self.nm :]
        else:
            box, cls = x_cat.split((self.nm, self.nc), 1)

        if self.export:
            box_value=F.interpolate(box.unsqueeze(1),scale_factor=1).squeeze(1)
            cls=F.interpolate(cls.unsqueeze(1),scale_factor=1).squeeze(1)
            return (box_value, cls)
            # return (box_value, cls)

        dbox = box * self.strides
        # print(self.strides)
        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return (y, x)

    def bias_init(self):
        prior_prob = 0.01
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, s in zip(m.cv2, m.stride):  # from
        #     # a[-1].bias.data[:36] = 0.  # cls (.01 objects, 80 classes, 640 img)
        #     # a[-1].bias.data[36:] = math.log(5 / m.nc / (128 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
            a[-1].bias.data[36:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
            a[-1].bias.data[:36] = 1.0  # box

            # a[-1].bias.data[36:]=-math.log((1 - prior_prob) / prior_prob)
            # a[-1].weight.data[:]=0.
            # a[-1].bias.data[:36]=0.  # box
            # init.normal_(a[-1].weight.data)
        

        # for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
        #     a[-1].bias.data[:36] = 1.0  # box
        #     b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class polarpaperDetect(nn.Module):
    # polarpaper v8标签分配
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):  # detection layer

        super().__init__()
        self.nm=36
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.nm  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, self.nm, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.nm]
            cls = x_cat[:, self.nm :]
        else:
            box, cls = x_cat.split((self.nm, self.nc), 1)

        if self.export:
            box_value=F.interpolate(box.unsqueeze(1),scale_factor=1).squeeze(1)
            cls=F.interpolate(cls.unsqueeze(1),scale_factor=1).squeeze(1)
            return (box_value, cls)
            # return (box_value, cls)

        dbox = box * self.strides
        # print(self.strides)
        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return (y, x)

    def bias_init(self):
        prior_prob = 0.01
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:36] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class Segment(Detect):
    """YOLOv8 Segment head for segmentation models."""
    #polarpaper  v8标签分配
    def __init__(self, nc=80, nm=36, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.detect = Detect.forward

    def make_anchors(self,feats, strides, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points, stride_tensor = [], []
        assert feats is not None
        dtype, device = feats[0].dtype, feats[0].device
        # print(strides)
        for i, stride in enumerate(strides):
            _,_, h, w = feats[i].shape
            # print(h,w)
            sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
            sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_points), torch.cat(stride_tensor)
    
    def distance2mask(self,points,distances,stride_tensor):
        B,C,_,_=distances[0].shape
        distances = torch.cat([xi.view(B, self.no, -1) for xi in distances], 2)

        B,C,an=distances.shape
        angles = torch.arange(0, 360, 360//36).type_as(distances)/180.*np.pi
        sin = torch.sin(angles)
        cos = torch.cos(angles)

        #B,C,H,W 2 B,C,an
        distances = distances.permute(0,2,1)#B,5+36,an  B,an,36+5
        distances_,cls =distances.split((self.nm, self.nc), -1)#B,an,36,B,an,5####################

        cls=cls.sigmoid()
        points = points.view(1,an,1,2).expand(B,-1,36,-1) #an, 2  -> B,an, 36,2
        stride_tensor = stride_tensor.view(1,an,1)#an, 1  -> B,an, 1
        # distances=distances.exp()*stride_tensor.clamp(min=1e-6)
        # distances=(distances*stride_tensor).exp().clamp(min=1e-6)
        distances_=(distances_*stride_tensor).clamp(min=1e-6)

        # select_bool=distances.gt(1e-6)
        select_bool=distances_.gt(1)
        c_xx, c_yy = points[:,:,:,0], points[:,:,:,1]#B,an,36
        segx = distances_ * cos + c_xx
        segy = distances_ * sin + c_yy #B,an,36
        seg=torch.cat((segx,segy,select_bool),-1)#B,an,36*2+1

        max_y=segy.max(-1)[0]
        min_y=segy.min(-1)[0]
        max_x=segx.max(-1)[0]
        min_x=segx.min(-1)[0]
        boxpred=torch.stack([min_x,min_y,max_x,max_y],-1)

        allpred=torch.cat((boxpred,cls, seg),-1).permute(0,2,1)




        # selectcenter=points[cls.max(-1)[0]>0.5]
        # selectdis=distances_[cls.max(-1)[0]>0.5]
        # if selectcenter.shape[0]==0:
        #     return allpred
        # # print('selectcenter.shape',selectcenter.shape)
        # # print('selectdis.shape',selectdis.shape)
        # # print(selectdis.shape)
        # # print(cls.max(-1)[0].shape)
        # # print(points.shape)
        # cxx,cyy = selectcenter[:,:,0],selectcenter[:,:,1]#B,an,36
        # sx =cxx+selectdis * cos
        # sy =cyy+selectdis *sin #B,an,36
        # segsxsy=torch.stack((sx,sy),-1)
        # disbool=selectdis.gt(1)
        # random_float_np =np.random.rand()
        # import cv2
        # for i in range(segsxsy.shape[0]):
        #     c_x = cxx[i][0].detach().cpu().numpy().astype(np.int32)
        #     c_y = cxx[i][0].detach().cpu().numpy().astype(np.int32)
        #     contour = segsxsy[i][disbool[i]].detach().cpu().numpy().astype(np.int32)
        #     img = np.zeros((self.imgh, self.imgw), dtype=np.uint8)
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
        #     cv2.imwrite('/home/data2/mxl/ultralytics/ultralytics-main/bb/bb1/sss{}_{}.jpg'.format(i,random_float_np), img)
        # # print(l)







        return allpred

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        if self.export==False:
            if self.training:
                x = self.detect(self, x)#(y,x)
                x = [xi for xi in x]
                return x,5,2
            else:
                x= self.detect(self, x)#eval
                feats = [xi for xi in x[1]]

                x = list(x)
                x = [xi for xi in x]
                anchor_points, stride_tensor=self.make_anchors(feats, self.stride)        
                self.imgh=feats[0].shape[2]*8
                self.imgw=feats[0].shape[3]*8
                allpred=self.distance2mask(anchor_points*stride_tensor,feats,stride_tensor)
                x = tuple(x)
            return ((allpred), (x[1],allpred,1))#eval

        if self.export:
            box_value,clss = self.detect(self, x)
            return (box_value, clss)
            
class polarSegment(Detect):
    """YOLOv8 Segment head for segmentation models."""
    #polar focos标签分配
    def __init__(self, nc=80, nm=36, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.detect = Detect.forward

    def make_anchors(self,feats, strides, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points, stride_tensor = [], []
        assert feats is not None
        dtype, device = feats[0].dtype, feats[0].device
        # print(strides)
        for i, stride in enumerate(strides):
            _,_, h, w = feats[i].shape
            # print(h,w)
            sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
            sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_points), torch.cat(stride_tensor)
    
    def distance2mask(self,points,distances,stride_tensor):
        random_float_np =np.random.rand()
        B,C,_,_=distances[0].shape
        distances = torch.cat([xi.view(B, self.no, -1) for xi in distances], 2)

        B,C,an=distances.shape
        angles = torch.arange(0, 360, 360//36).type_as(distances)/180.*np.pi
        sin = torch.sin(angles)
        cos = torch.cos(angles)

        #B,C,H,W 2 B,C,an
        distances_ = distances.permute(0,2,1)#B,5+36,an  B,an,36+5
        distances,cls =distances_.split((self.nm, self.nc), -1)#B,an,36,B,an,5####################
        cls=cls.sigmoid()
        cls[...,-1]=0.0 
        centerness=distances[:,:,36:]
        centerness=centerness.sigmoid()
        cls*=centerness
        distances=distances[:,:,:36]
        points = points.view(1,an,1,2).expand(B,-1,36,-1) #an, 2  -> B,an, 36,2
        stride_tensor = stride_tensor.view(1,an,1)#an, 1  -> B,an, 1
        # distances=distances.exp()*stride_tensor.clamp(min=1e-6)
        # distances=(distances*stride_tensor).exp().clamp(min=1e-6)
        distances=(distances*stride_tensor).clamp(min=1e-6)

        # print(distances)
        # select_b=cls.max(-1)[0]>0.1
        # select_dis=distances[cls.max(-1)[0]>0.8]   
        # select_center=points[cls.max(-1)[0]>0.8]
        # # print(cls.max(-1)[0][cls.max(-1)[0]>0.1].shape)

        # # print(points[cls.max(-1)[0]>0.1].shape)
        # dis_bool=select_dis.gt(1)
        # c_xx, c_yy = select_center[:,:,0], select_center[:,:,1]#B,an,36
        # segx = c_xx+select_dis * cos 
        # segy = c_yy+select_dis * sin  #B,an,36
        # # print(segx.shape)

        # seg=torch.stack((segx,segy),-1)
        # print(seg.shape)
        # print(seg[-1])
        # print(select_center[-1])
        # print(cos[-1])
        # print(sin[-1])
        # print(select_dis[-1])
        # print(segx[-1],segy[-1])

        # import cv2
        # for i in range(seg.shape[0]):
        #     c_x = c_xx[i][0].detach().cpu().numpy().astype(np.int32)
        #     c_y = c_yy[i][0].detach().cpu().numpy().astype(np.int32)

        #     contour = seg[i][dis_bool[i]].detach().cpu().numpy().astype(np.int32)
        #     img = np.zeros((self.imgh, self.imgw), dtype=np.uint8)
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
        #     cv2.imwrite('/home/data2/mxl/ultralytics/ultralytics-main/bb/bb1/sss{}_{}.jpg'.format(i,random_float_np), img)
        # print(l)
        # print(l)

        # print(distances)
        select_bool=distances.gt(1e-5)
        c_x, c_y = points[:,:,:,0], points[:,:,:,1]#B,an,36
        segx = distances * cos + c_x
        segy = distances * sin + c_y #B,an,36
        seg=torch.cat((segx,segy,select_bool),-1)#B,an,36*2+1
        max_y=segy.max(-1)[0]
        min_y=segy.min(-1)[0]
        max_x=segx.max(-1)[0]
        min_x=segx.min(-1)[0]
        # print(max_y.shape)
        # print(torch.stack([min_x,min_y,max_x,max_y],-1).shape)
        boxpred=torch.stack([min_x,min_y,max_x,max_y],-1)

        allpred=torch.cat((boxpred,cls, seg),-1).permute(0,2,1)
        # print('allpred.shape',allpred.shape)

        #x=B,C, An
        return allpred

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        # for xi in x:
        #     print(xi.shape)
        if self.export==False:
            if self.training:
                x = self.detect(self, x)#(y,x)
                x = [xi for xi in x]
                return x,5,2
            else:
                x= self.detect(self, x)#eval
                feats = [xi for xi in x[1]]

                x = list(x)
                x = [xi for xi in x]
                # print(len(feats))
                # print(feats[0].shape)

                anchor_points, stride_tensor=self.make_anchors(feats, self.stride)
                
                self.imgh=feats[0].shape[2]*8
                self.imgw=feats[0].shape[3]*8
                # print(feats[0].shape[2],feats[0].shape[3])
                allpred=self.distance2mask(anchor_points*stride_tensor,feats,stride_tensor)
                x = tuple(x)
            return ((allpred), (x[1],allpred,1))#eval

        if self.export:
            box_value,clss = self.detect(self, x)
            return (box_value, clss)
        
class Segmentori(Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        if self.export:
            box_value,clss = self.detect(self, x)
            return (box_value, clss,mc, p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))

        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))

        



class Pose(Detect):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3].sigmoid_()  # inplace sigmoid
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        c_ = 512  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        # return x if self.training else x.softmax(1)
        return x.sigmoid() if self.training else x.sigmoid()


class RTDETRDecoder(nn.Module):
    export = False  # export mode

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hd=256,  # hidden dim
            nq=300,  # num queries
            ndp=4,  # num decoder points
            nh=8,  # num head
            ndl=6,  # num decoder layers
            d_ffn=1024,  # dim of feedforward
            dropout=0.,
            act=nn.ReLU(),
            eval_idx=-1,
            # training args
            nd=100,  # num denoising
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False):
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        from ultralytics.models.utils.ops import get_cdn_group

        # input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_cdn_group(batch,
                          self.nc,
                          self.num_queries,
                          self.denoising_class_embed.weight,
                          self.num_denoising,
                          self.label_noise_ratio,
                          self.box_noise_scale,
                          self.training)

        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # decoder
        dec_bboxes, dec_scores = self.decoder(embed,
                                              refer_bbox,
                                              feats,
                                              shapes,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask)
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        anchors = []
        for i, (h, w) in enumerate(shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(end=h, dtype=dtype, device=device),
                                            torch.arange(end=w, dtype=dtype, device=device),
                                            indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([h, w], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float('inf'))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        # get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        bs = len(feats)
        # prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init_` would cause NaN when training with custom datasets.
        # linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)
