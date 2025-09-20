import torch
import torch.nn as nn
import numpy as np
import sys
import colorsys
import os

import cv2
from PIL import Image
from PIL import ImageDraw, ImageFont
from torchvision.ops import nms
import pkg_resources as pkg
# import src.sort as sort
from src.V4R_sv.Line.Color_list import Color
from src.V4R_sv.Line.Line_counter import LineZone, LineZoneAnnotator
from src.V4R_sv.Line.Annotor import BoxAnnotator
from src.V4R_sv.Line.detectcore import Detections
from src.V4R_sv.Line.gemcore import Point
from src.V4R_sv.tracker.byte_tracker.core import ByteTrack
# from src.V4R_sv.volume_estimate.utils import getContours_box, getContours_cont, object_cut
from src.V4R_sv.Line.counting_assist import counting_assist
# from src.V4R_sv.det_draw import draw_bounding_box_on_image

#---------------------------------------------------------#
#   yolo_utils<utils.py>
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

# def resize_image(image, size, letterbox_image):
#     iw, ih  = image.size
#     w, h    = size
#     if letterbox_image:
#         scale   = min(w/iw, h/ih)
#         nw      = int(iw*scale)
#         nh      = int(ih*scale)
#
#         image   = image.resize((nw,nh), Image.BICUBIC)
#         new_image = Image.new('RGB', size, (128,128,128))
#         new_image.paste(image, ((w-nw)//2, (h-nh)//2))
#     else:
#         new_image = image.resize((w, h), Image.BICUBIC)
#     return new_image

def resize_image(img, size, letterbox_image):
    height, width, channels = img.shape
    if letterbox_image:
        # 如果图片宽度大于高度，则添加灰色矩形条
        if width > height:
            # 创建一个新的黑色画布
            canvas = np.zeros((width, width, channels), dtype=np.uint8)
            # 将原始图片复制到新画布中间
            canvas[int((width - height) / 2):int((width + height) / 2), :, :] = img
            # 在上下两侧添加灰色矩形条
            canvas[0:int((width - height) / 2), :, :] = 128
            canvas[int((width + height) / 2):width, :, :] = 128
            # 调整图像大小为640*640
            resized = cv2.resize(canvas, size)
        elif width < height:
            # 创建一个新的黑色画布
            canvas = np.zeros((height, height, channels), dtype=np.uint8)
            # 将原始图片复制到新画布中间
            canvas[:, int((height - width) / 2):int((width + height) / 2), :] = img
            # 在上下两侧添加灰色矩形条
            canvas[:, 0:int((height - width) / 2), :] = 128
            canvas[:, int((height + width) / 2):height, :] = 128
            # 调整图像大小为640*640
            resized = cv2.resize(canvas, size)
        else:
            # 图片已经是正方形，不需要进行处理
            resized = cv2.resize(img, size)
    else:
        resized = cv2.resize(img, size)
    return resized

def preprocess_input(image):
    image /= 255.0
    return image

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def wh2c(l, t, w, h):
    cx = l + w // 2
    cy = t + h // 2
    return cx, cy



# ---------------------------------------------------#
#   yolo_decode<utils_bbox.py>
# ---------------------------------------------------#

def check_version(current: str = "0.0.0",
                  minimum: str = "0.0.0",
                  name: str = "version ",
                  pinned: bool = False) -> bool:
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    return result

TORCH_1_10 = check_version(torch.__version__, '1.10.0')

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w  = feats[i].shape
        sx          = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy          = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx      = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    # 左上右下
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

class DecodeBox():
    def __init__(self, num_classes, input_shape):
        super(DecodeBox, self).__init__()
        self.num_classes = num_classes
        self.bbox_attrs = 4 + num_classes
        self.input_shape = input_shape

    def decode_box(self, inputs):
        # dbox  batch_size, 4, 8400
        # cls   batch_size, 20, 8400
        dbox, cls, origin_cls, anchors, strides = inputs
        # 获得中心宽高坐标
        dbox = dist2bbox(dbox, anchors.unsqueeze(0), xywh=True, dim=1) * strides
        y = torch.cat((dbox, cls.sigmoid()), 1).permute(0, 2, 1)
        # 进行归一化，到0~1之间
        y[:, :, :4] = y[:, :, :4] / torch.Tensor(
            [self.input_shape[1], self.input_shape[0], self.input_shape[1], self.input_shape[0]]).to(y.device)
        return y

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        # -----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        # -----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            # -----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            # -----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
                               axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5,
                            nms_thres=0.4):
        # ----------------------------------------------------------#
        #   将预测结果的格式转换成左上角右下角的格式。
        #   prediction  [batch_size, num_anchors, 85]
        # ----------------------------------------------------------#
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            # ----------------------------------------------------------#
            #   对种类预测部分取max。
            #   class_conf  [num_anchors, 1]    种类置信度
            #   class_pred  [num_anchors, 1]    种类
            # ----------------------------------------------------------#
            class_conf, class_pred = torch.max(image_pred[:, 4:4 + num_classes], 1, keepdim=True)

            # ----------------------------------------------------------#
            #   利用置信度进行第一轮筛选
            # ----------------------------------------------------------#
            conf_mask = (class_conf[:, 0] >= conf_thres).squeeze()

            # ----------------------------------------------------------#
            #   根据置信度进行预测结果的筛选
            # ----------------------------------------------------------#
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not image_pred.size(0):
                continue
            # -------------------------------------------------------------------------#
            #   detections  [num_anchors, 6]
            #   6的内容为：x1, y1, x2, y2, class_conf, class_pred
            # -------------------------------------------------------------------------#
            detections = torch.cat((image_pred[:, :4], class_conf.float(), class_pred.float()), 1)

            # ------------------------------------------#
            #   获得预测结果中包含的所有种类
            # ------------------------------------------#
            unique_labels = detections[:, -1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            for c in unique_labels:
                # ------------------------------------------#
                #   获得某一类得分筛选后全部的预测结果
                # ------------------------------------------#
                detections_class = detections[detections[:, -1] == c]
                # ------------------------------------------#
                #   使用官方自带的非极大抑制会速度更快一些！
                #   筛选出一定区域内，属于同一种类得分最大的框
                # ------------------------------------------#
                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4],
                    nms_thres
                )
                max_detections = detections_class[keep]

                # # 按照存在物体的置信度排序
                # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
                # detections_class = detections_class[conf_sort_index]
                # # 进行非极大抑制
                # max_detections = []
                # while detections_class.size(0):
                #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
                #     max_detections.append(detections_class[0].unsqueeze(0))
                #     if len(detections_class) == 1:
                #         break
                #     ious = bbox_iou(max_detections[-1], detections_class[1:])
                #     detections_class = detections_class[1:][ious < nms_thres]
                # # 堆叠
                # max_detections = torch.cat(max_detections).data

                # Add max detections to outputs
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output


# ---------------------------------------------------#
#   yolo_backbone<backbone.py>
# ---------------------------------------------------#

def autopad(k, p=None, d=1):
    # kernel, padding, dilation
    # 对输入的特征层进行自动padding，按照Same原则
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class SiLU(nn.Module):
    # SiLU激活函数
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class Conv(nn.Module):
    # 标准卷积+标准化+激活函数
    default_act = SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # 标准瓶颈结构，残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    # CSPNet结构结构，大残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c      = int(c2 * e)
        self.cv1    = Conv(c1, 2 * self.c, 1, 1)
        self.cv2    = Conv((2 + n) * self.c, c2, 1)
        self.m      = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # 进行一个卷积，然后划分成两份，每个通道都为c
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    # SPP结构，5、9、13最大池化核的最大池化。
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_          = c1 // 2
        self.cv1    = Conv(c1, c_, 1, 1)
        self.cv2    = Conv(c_ * 4, c2, 1, 1)
        self.m      = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Backbone(nn.Module):
    def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
        super().__init__()
        # -----------------------------------------------#
        #   输入图片是3, 640, 640
        # -----------------------------------------------#
        # 3, 640, 640 => 32, 640, 640 => 64, 320, 320
        self.stem = Conv(3, base_channels, 3, 2)

        # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C2f(base_channels * 2, base_channels * 2, base_depth, True),
        )
        # 128, 160, 160 => 256, 80, 80 => 256, 80, 80)
        # self.img_cap.setEnabled(False)
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        )
        # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
        )
        # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)
        )

        if pretrained:
            url = {
                "n": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
                "s": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
                "m": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
                "l": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
                "x": 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        # -----------------------------------------------#
        #   dark3的输出为256, 80, 80，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        # -----------------------------------------------#
        #   dark4的输出为512, 40, 40，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        # -----------------------------------------------#
        #   dark5的输出为1024 * deep_mul, 20, 20，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3

# ---------------------------------------------------#
#   yolo_body<yolo.py>
# ---------------------------------------------------#

def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    # print('initialize network with %s type' % init_type)
    net.apply(init_func)

def fuse_conv_and_bn(conv, bn):
    # 混合Conv2d + BatchNorm2d 减少计算量
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # 准备kernel
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # 准备bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

class DFL(nn.Module):
    # DFL模块
    # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv   = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x           = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1     = c1

    def forward(self, x):
        # bs, self.reg_max * 4, 8400
        b, c, a = x.shape
        # bs, 4, self.reg_max, 8400 => bs, self.reg_max, 4, 8400 => b, 4, 8400
        # 以softmax的方式，对0~16的数字计算百分比，获得最终数字。
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

class YoloBody(nn.Module):
    def __init__(self, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()

        depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.00, }
        width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        deep_width_dict = {'n': 1.00, 's': 1.00, 'm': 0.75, 'l': 0.50, 'x': 0.50, }
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # -----------------------------------------------#
        #   输入图片是3, 640, 640
        # -----------------------------------------------#

        # ---------------------------------------------------#
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   256, 80, 80
        #   512, 40, 40
        #   1024 * deep_mul, 20, 20
        # ---------------------------------------------------#
        self.backbone = Backbone(base_channels, base_depth, deep_mul, phi, pretrained=pretrained)

        # ------------------------加强特征提取网络------------------------#
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        self.conv3_for_upsample1 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, base_channels * 8,
                                       base_depth, shortcut=False)
        # 768, 80, 80 => 256, 80, 80
        self.conv3_for_upsample2 = C2f(base_channels * 8 + base_channels * 4, base_channels * 4, base_depth,
                                       shortcut=False)

        # 256, 80, 80 => 256, 40, 40
        self.down_sample1 = Conv(base_channels * 4, base_channels * 4, 3, 2)
        # 512 + 256, 40, 40 => 512, 40, 40
        self.conv3_for_downsample1 = C2f(base_channels * 8 + base_channels * 4, base_channels * 8, base_depth,
                                         shortcut=False)

        # 512, 40, 40 => 512, 20, 20
        self.down_sample2 = Conv(base_channels * 8, base_channels * 8, 3, 2)
        # 1024 * deep_mul + 512, 20, 20 =>  1024 * deep_mul, 20, 20
        self.conv3_for_downsample2 = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8,
                                         int(base_channels * 16 * deep_mul), base_depth, shortcut=False)
        # ------------------------加强特征提取网络------------------------#

        ch = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
        self.shape = None
        self.nl = len(ch)
        # self.stride     = torch.zeros(self.nl)
        self.stride = torch.tensor(
            [256 / x.shape[-2] for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))])  # forward
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = num_classes + self.reg_max * 4  # number of outputs per anchor
        self.num_classes = num_classes

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, num_classes, 1)) for x in ch)
        if not pretrained:
            weights_init(self)

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def fuse(self):
        # print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        return self

    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone.forward(x)

        # ------------------------加强特征提取网络------------------------#
        # 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 40, 40
        P5_upsample = self.upsample(feat3)
        # 1024 * deep_mul, 40, 40 cat 512, 40, 40 => 1024 * deep_mul + 512, 40, 40
        P4 = torch.cat([P5_upsample, feat2], 1)
        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        P4 = self.conv3_for_upsample1(P4)

        # 512, 40, 40 => 512, 80, 80
        P4_upsample = self.upsample(P4)
        # 512, 80, 80 cat 256, 80, 80 => 768, 80, 80
        P3 = torch.cat([P4_upsample, feat1], 1)
        # 768, 80, 80 => 256, 80, 80
        P3 = self.conv3_for_upsample2(P3)

        # 256, 80, 80 => 256, 40, 40
        P3_downsample = self.down_sample1(P3)
        # 512, 40, 40 cat 256, 40, 40 => 768, 40, 40
        P4 = torch.cat([P3_downsample, P4], 1)
        # 768, 40, 40 => 512, 40, 40
        P4 = self.conv3_for_downsample1(P4)

        # 512, 40, 40 => 512, 20, 20
        P4_downsample = self.down_sample2(P4)
        # 512, 20, 20 cat 1024 * deep_mul, 20, 20 => 1024 * deep_mul + 512, 20, 20
        P5 = torch.cat([P4_downsample, feat3], 1)
        # 1024 * deep_mul + 512, 20, 20 => 1024 * deep_mul, 20, 20
        P5 = self.conv3_for_downsample2(P5)
        # ------------------------加强特征提取网络------------------------#
        # P3 256, 80, 80
        # P4 512, 40, 40
        # P5 1024 * deep_mul, 20, 20
        shape = P3.shape  # BCHW

        # P3 256, 80, 80 => num_classes + self.reg_max * 4, 80, 80
        # P4 512, 40, 40 => num_classes + self.reg_max * 4, 40, 40
        # P5 1024 * deep_mul, 20, 20 => num_classes + self.reg_max * 4, 20, 20
        x = [P3, P4, P5]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        
        return x, shape

        # if self.shape != shape:
        #     self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        #     self.shape = shape

        # # num_classes + self.reg_max * 4 , 8400 =>  cls num_classes, 8400;
        # #                                           box self.reg_max * 4, 8400
        # box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split(
        #     (self.reg_max * 4, self.num_classes), 1)
        # # origin_cls      = [xi.split((self.reg_max * 4, self.num_classes), 1)[1] for xi in x]
        # dbox = self.dfl(box)
        # return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device)

class yolobody_trt(YoloBody):
    def __init__(self, num_classes, phi):
        super(yolobody_trt, self).__init__(num_classes=num_classes, phi=phi)

    def forward(self, x, shape):
        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        # num_classes + self.reg_max * 4 , 8400 =>  cls num_classes, 8400;
        #                                           box self.reg_max * 4, 8400
        # box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split(
        #     (self.reg_max * 4, self.num_classes), 1)
        box, cls = torch.cat([xi.view(1, self.no, -1) for xi in x], 2).split(
            (self.reg_max * 4, self.num_classes), 1)
        # origin_cls      = [xi.split((self.reg_max * 4, self.num_classes), 1)[1] for xi in x]
        dbox = self.dfl(box)
        return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device)



# 搭建框架
class YOLO(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        # "model_path"        : 'model_data/yolov8_s.pth',
        # "classes_path"      : 'model_data/coco_classes.txt',
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        # "input_shape"       : [640, 640],
        "input_shape": [320, 320],
        #------------------------------------------------------#
        #   所使用到的yolov8的版本：
        #   n : 对应yolov8_n
        #   s : 对应yolov8_s
        #   m : 对应yolov8_m
        #   l : 对应yolov8_l
        #   x : 对应yolov8_x
        #------------------------------------------------------#
        "phi"               : 's', # n
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.6,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.6,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, model_path, cls_path, speed_up: bool, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

        # 根据需要 添加的参数
        self.model_path = model_path
        self.classes_path = cls_path
        self.speed_up = speed_up


        # ---------------------------------------------------#
        #   画线检测初始化
        # ---------------------------------------------------#
        self.line_counter = LineZone(start=Point(900, 0), end=Point(900, 1080)) #设定越线检测位置
        self.line_annotator = LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=2, color=Color(r=224, g=57, b=151))
        # self.tracker = sort.Sort(max_age = 90, min_hits= 15 , 
        #                     iou_threshold = 0.3)
        self.tracker = ByteTrack()
        self.depth_dict = {}
        self.fish_num = 0
        self.fish_id = []
        self.fish_l = []
        self.fish_w = []

        # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.bbox_util = DecodeBox(self.num_classes, (self.input_shape[0], self.input_shape[1]))

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

        # show_config(**self._defaults)

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self, onnx=False):
        # ---------------------------------------------------#
        #   建立yolo模型，载入yolo模型的权重
        # ---------------------------------------------------#
        '''
        self.net = YoloBody(self.anchors_mask, self.num_classes, self.phi)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        # print('{} model, and classes {} loaded.'.format(self.model_path, self.classes_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()
        '''
        if sys.platform == "linux":
            if self.speed_up:
                from torch2trt import TRTModule
                self.net1 = TRTModule()
                self.net2 = yolobody_trt(self.num_classes, self.phi)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                self.net1.load_state_dict(torch.load('src/det_cfg/fishbest_solo_s.pth'))
                self.net2.load_state_dict(torch.load(self.model_path, map_location=device)) 
                self.net2 = self.net2.fuse().eval()

                if not onnx:
                    if self.cuda:
                        # self.net1 = nn.DataParallel(self.net1).cuda()
                        self.net2 = nn.DataParallel(self.net2).cuda()


            else:
                self.net1 = YoloBody(self.num_classes, self.phi)
                self.net2 = yolobody_trt(self.num_classes, self.phi)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                self.net1.load_state_dict(torch.load(self.model_path, map_location=device)) 
                self.net2.load_state_dict(torch.load(self.model_path, map_location=device)) 
                self.net1 = self.net1.fuse().eval()
                self.net2 = self.net2.fuse().eval()

                if not onnx:
                    if self.cuda:
                        self.net1 = nn.DataParallel(self.net1).cuda()
                        self.net2 = nn.DataParallel(self.net2).cuda()
                # self.net = YoloBody(self.input_shape, self.num_classes, self.phi)
                # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # self.net.load_state_dict(torch.load(self.model_path, map_location=device))
                # self.net = self.net.eval()
                # if not onnx:
                #     if self.cuda:
                #         self.net = nn.DataParallel(self.net)
                #         self.net = self.net.cuda()

        elif sys.platform == "win32":

            self.net1 = YoloBody(self.num_classes, self.phi)
            self.net2 = yolobody_trt(self.num_classes, self.phi)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            self.net1.load_state_dict(torch.load(self.model_path, map_location=device)) # 加载模型错误
            self.net2.load_state_dict(torch.load(self.model_path, map_location=device)) # 加载模型错误
            self.net1 = self.net1.fuse().eval()
            self.net2 = self.net2.fuse().eval()

            if not onnx:
                if self.cuda:
                    self.net1 = nn.DataParallel(self.net1).cuda()
                    self.net2 = nn.DataParallel(self.net2).cuda()
        else:
            raise EnvironmentError("the platform must be win32 or linux(Nvidia NX).")
            

    def detect_tensor(self, frame, depth_frame=None, crop=False, count=False):
        """
        重写原有 detect_image 函数，主要考虑与界面接口协调问题
        """
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        frame_h, frame_w = np.array(np.shape(frame)[0:2])
        # print(frame_h)
        # print(frame_w)
        frame_cx ,frame_cy = wh2c(0, 0, frame_w, frame_h)
        shape_l = frame_cx - 50
        shape_r = frame_cx + 50

        # 深度计数校正相关
        depth_l = 259 
        depth_t = 36 
        depth_r = 886 
        depth_b = 506
        depth_w = depth_r - depth_l
        depth_h = depth_b - depth_t
        depth_region = [depth_l, depth_t, depth_r, depth_b] 
        depth_rcx, depth_rcy = wh2c(depth_l, depth_t, depth_w, depth_h)
        cont_l = depth_rcx - 20 # change5
        cont_r = depth_rcx + 20
        multi_fish = False

        # ---------------------------------------------------------#
        #   先变换通道顺序，再将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cvtColor(image)

        # image = cvtColor(frame)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        #   h, w, 3 => 3, h, w => 1, 3, h, w
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #   将图像输入网络当中进行预测！
            outputs0, s = self.net1(images)  #here 获取检测框
            outputs = self.net2(outputs0, s)
            outputs = self.bbox_util.decode_box(outputs)
            #   将预测框进行堆叠，然后进行非极大抑制
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape,
                                                         (frame_h, frame_w), self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)
        if results[0] is None:#添加相应跟踪逻辑
            detections = np.empty((0, 9))
            detections = Detections.from_yolov4r(detections)
            detections = self.tracker.update_with_detections(detections)

            self.line_counter.trigger(detections=detections)
            # self.line_annotator.annotate(frame=frame, line_counter=self.line_counter) #change 不画线
            self.fish_num = self.line_counter.out_count - self.line_counter.in_count

            # show_num = {self.fish_num}
            # show_num = show_num.zfill(4)

            count_txt = f"{self.fish_num}".zfill(5)
            frame = cv2.putText(frame, count_txt, (15,85), 0, 3, (255, 0, 0), 10)

            maxl_dict= self.tracker.max_lengths
            maxw_dict= self.tracker.max_widths
            self.fish_id = list(maxl_dict.keys())  
            self.fish_l = list(maxl_dict.values())
            self.fish_w = list(maxw_dict.values())
            # self.fish_lw = list(zip(fish_l, fish_w))

            # return frame, self.fish_num, self.fish_id, self.fish_lw
            return frame
        
        # return results
        top_results = results[0]
        top_label = np.array(results[0][:, 5], dtype='int32')
        top_conf = results[0][:, 4]
        top_boxes = results[0][:, :4]

        thickness = int(max((frame_w + frame_h) // 640, 1))
        # thickness = 2
        # thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        # ---------------------------------------------------------#
        #   计数
        # ---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        # ---------------------------------------------------------#
        #   是否进行目标的裁剪
        # ---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(frame_h, np.floor(bottom).astype('int32'))
                right = min(frame_w, np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        # ---------------------------------------------------------#
        #   是否进行目标轮廓检测
        # ------------------------------------ret, self.img = self.cap.read()---------------------#
        """ if cont:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(frame_h, np.floor(bottom).astype('int32'))
                right = min(frame_w, np.floor(right).astype('int32'))

                xyxy = [left, top, right, bottom]
                cut_img = object_cut(frame, xyxy)

                # # 直接展示轮廓
                # contours, area = getContours_cont(cut_img)
                # frame = cv2.drawContours(frame, [contours], 0, (0, 255, 0), 3)

                # 最小矩形拟合
                box, area, rect = getContours_box(cut_img)
                m, n = box.shape
                for i in range(m):
                    box[i][0] = box[i][0] + left
                    box[i][1] = box[i][1] + top 
                frame = cv2.polylines(frame, [box], True, (0, 255, 0), 3) """

        
        
        currentArray = np.empty((0, 9))
        detections = np.empty((0, 9))
        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            # 获取类别标签
            predicted_class = self.class_names[int(c)]
            class_index = int(c) 
            # 边框与置信度
            box = top_boxes[i]
            score = top_conf[i]

            # 计算边框坐标值并圆整
            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(frame_h, np.floor(bottom).astype('int32'))
            right = min(frame_w, np.floor(right).astype('int32'))

            # # 获得检测轮廓
            # xyxy_cont = [left, top, right, bottom]
            # cut_img = object_cut(frame, xyxy_cont)
            # box, area, rect = getContours_box(cut_img)
            # width, length = rect
            # width = int(width)
            # length = int(length)
            # if length < width:
            #     temp = length
            #     length = width
            #     width = temp
            # area = int(area)

            # 获取检测框中心点  #change2
            d_l = right - left
            d_w = bottom - top
            cx, cy = wh2c(left, top, d_l, d_w)
            if cx >= shape_l and cx <= shape_r:
                # 获得检测轮廓
                # xyxy_cont = [left, top, right, bottom]
                # cut_img = object_cut(frame, xyxy_cont)
                # box, area, rect = getContours_box(cut_img)
                length = d_l
                width = d_w
                
                width = int(width)
                length = int(length)  
            else:
                width = 0
                length = 0

            #画框
            k1 = (left, top)
            k2 = (right, bottom)
            p1 = (left, top)
            p2 = (right, bottom)
            label = '{}'.format(predicted_class)

            # 标签
            tf = max(thickness - 2, 1)  # font thickness
            w1, h1 = cv2.getTextSize(label, 0, fontScale=thickness / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h1 >= 3
            p2 = p1[0] + w1, p1[1] - h1 - 3 if outside else p1[1] + h1 + 3

            # 获取检测框中心点  #change1
            # tracked_cx, tracked_cy = wh2c(x1, y1, x2 - x1, y2 - y1)
            # if (tracked_cx >= cont_l) and (tracked_cx <= cont_r)
            if (cx >= cont_l) and (cx <= cont_r):
                multi_fish = counting_assist(depth_frame, [left,top,right,bottom])
            # if multi_fish:
            #     if not tracker_id in self.depth_dict:
            #         self.depth_dict[tracker_id] = 1
            #     else:
            #         self.depth_dict[tracker_id] += 1
            if multi_fish:
                # 目标框
                cv2.rectangle(frame, k1, k2, (0,255,255), thickness=thickness, lineType=cv2.LINE_AA)
                
                # 标签矩形
                cv2.rectangle(frame, p1, p2, (0,255,255), -1, cv2.LINE_AA)  # filled
                frame = cv2.putText(frame,
                            label, (p1[0], p1[1] - 2 if outside else p1[1] + h1 + 2),
                            0,
                            thickness / 3,
                            (0, 0, 0),
                            thickness=tf,
                            lineType=cv2.LINE_AA)

                # frame = draw_bounding_box_on_image(frame,top,left,bottom,right,color='yellow')
                # 深度校正
                m_fish = 1

            else:
                # 目标框
                cv2.rectangle(frame, k1, k2, self.colors[c], thickness=thickness, lineType=cv2.LINE_AA)
                # print(frame.size)
                
                # 标签矩形
                cv2.rectangle(frame, p1, p2, self.colors[c], -1, cv2.LINE_AA)  # filled
                frame = cv2.putText(frame,
                            label, (p1[0], p1[1] - 2 if outside else p1[1] + h1 + 2),
                            0,
                            thickness / 3,
                            (255, 255, 255),
                            thickness=tf,
                            lineType=cv2.LINE_AA)

                # frame = draw_bounding_box_on_image(frame,top,left,bottom,right,color='blue')

                m_fish = 0
            
            currentArray = np.array([left, top, right, bottom, score, class_index, length, width, m_fish])
            detections = np.vstack((detections, currentArray))
        
        detections = Detections.from_yolov4r(detections)
        detections = self.tracker.update_with_detections(detections)


# change 不显示ID,去掉第二个for循环
        
        for detection in detections:
            xyxy,_,_,_,tracker_id, length, width, m_fish = detection
            tracker_id = str(tracker_id)
            # x1,y1,x2,y2 = xyxy
            # x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # y1 = max(0, np.floor(y1).astype('int32'))
            # x1 = max(0, np.floor(x1).astype('int32'))
            # y2 = min(frame_h, np.floor(y2).astype('int32'))
            # x2 = min(frame_w, np.floor(x2).astype('int32'))

            # 获取检测框中心点  #change1

            if m_fish:
                if not tracker_id in self.depth_dict:
                    self.depth_dict[tracker_id] = 1
                else:
                    self.depth_dict[tracker_id] += 1

            # #画框
            # k1 = (x1, y1)
            # k2 = (x2, y2)
            # p1 = (x1, y1)
            # p2 = (x2, y2)
            # label = '{}'.format(predicted_class)

            # # 标签
            # tf = max(thickness - 2, 1)  # font thickness
            # w1, h1 = cv2.getTextSize(label, 0, fontScale=thickness / 3, thickness=tf)[0]  # text width, height
            # outside = p1[1] - h1 >= 3
            # p2 = p1[0] + w1, p1[1] - h1 - 3 if outside else p1[1] + h1 + 3

            # # ID
            # label_id = 'ID: {}'.format(tracker_id)
            # w2, h2 = cv2.getTextSize(label_id, 0, fontScale=thickness / 3, thickness=tf)[0]
            # p3 = (x2,y1)
            # p4 = (x2,y1)
            # outside = y1 - h2 >= 3
            # p4 = p3[0] - w2, y1 - h2 - 3 if outside else y1 + h2 + 3

            # #判断是否绘制警告框   #change:将class与id合在一起绘制，并完成警告框绘制逻辑
            # if m_fish:
            #     # 目标框
            #     cv2.rectangle(frame, k1, k2, (0,255,255), thickness=thickness, lineType=cv2.LINE_AA)
                
            #     # 标签矩形
            #     cv2.rectangle(frame, p1, p2, (0,255,255), -1, cv2.LINE_AA)  # filled
            #     frame = cv2.putText(frame,
            #                 label, (p1[0], p1[1] - 2 if outside else p1[1] + h1 + 2),
            #                 0,
            #                 thickness / 3,
            #                 (0, 0, 0),
            #                 thickness=tf,
            #                 lineType=cv2.LINE_AA)

                # # ID矩形
                # cv2.rectangle(frame, p3, p4, (0,255,255), -1, cv2.LINE_AA)
                # frame = cv2.putText(frame,
                #             label_id, (p4[0], y1 - 2 if outside else y1 + h2 + 2),
                #             0,
                #             thickness / 3,
                #             (0, 0, 0),
                #             thickness=tf,
                #             lineType=cv2.LINE_AA)
                
            # else:
            #     # 目标框
            #     cv2.rectangle(frame, k1, k2, self.colors[c], thickness=thickness, lineType=cv2.LINE_AA)
                
            #     # 标签矩形
            #     cv2.rectangle(frame, p1, p2, self.colors[c], -1, cv2.LINE_AA)  # filled
            #     frame = cv2.putText(frame,
            #                 label, (p1[0], p1[1] - 2 if outside else p1[1] + h1 + 2),
            #                 0,
            #                 thickness / 3,
            #                 (255, 255, 255),
            #                 thickness=tf,
            #                 lineType=cv2.LINE_AA)

                # # ID矩形
                # cv2.rectangle(frame, p3, p4, self.colors[c], -1, cv2.LINE_AA)
                # frame = cv2.putText(frame,
                #             label_id, (p4[0], y1 - 2 if outside else y1 + h2 + 2),
                #             0,
                #             thickness / 3,
                #             (255, 255, 255),
                #             thickness=tf,
                #             lineType=cv2.LINE_AA)


        # 将跟踪结果进行反馈计数
        self.line_counter.trigger(detections=detections) 
        # self.line_annotator.annotate(frame=frame, line_counter=self.line_counter)

        # 进行深度计数校正,该方法需要ID，故暂不使用
        depth_items = self.depth_dict.items()
        for key, value in depth_items:
            if value > 1:
                self.line_counter.out_count += 1
                self.depth_dict[key] = -9999
                # print('+1')
        
        self.fish_num = self.line_counter.out_count - self.line_counter.in_count

        count_txt = f"{self.fish_num}".zfill(5)
        frame = cv2.putText(frame, count_txt, (15,85), 0, 3, (255, 0, 0), 10)

        # 调取跟踪历程中各ID最大值字典
        maxl_dict= self.tracker.max_lengths
        maxw_dict= self.tracker.max_widths
        self.fish_id = list(maxl_dict.keys())  
        self.fish_l = list(maxl_dict.values())
        self.fish_w = list(maxw_dict.values())
        # self.fish_lw = list(zip(fish_l, fish_w))

        # return frame, self.fish_num, self.fish_id, self.fish_lw
        return frame
