# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
import scipy
import math
from scipy.spatial.distance import cdist

from ultralytics.utils.metrics import bbox_ioa

try:
    import lap  # for linear_assignment

    assert lap.__version__  # verify package is not directory
except (ImportError, AssertionError, AttributeError):
    from ultralytics.utils.checks import check_requirements

    check_requirements('lapx>=0.5.2')  # update to lap package from https://github.com/rathaROG/lapx
    import lap


def calculate_diagonal_length(tlwh):
    """
    根据左上角坐标和宽高信息（_tlwh格式）计算检测框的对角线长度
    Args:
        tlwh (list or tuple or np.ndarray): 包含左上角x坐标、左上角y坐标、宽度、高度的序列，格式为 [x, y, w, h]
    Returns:
        float: 检测框的对角线长度
    """
    width = tlwh[2]
    height = tlwh[3]
    return width / height

def GIoU(box1, box2):
    # 计算两个图像的最小外接矩形的面积
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    area_c = (max(x2, x4) - min(x1, x3)) * (max(y4, y2) - min(y3, y1))

    # 计算中间矩形的宽高
    in_w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_h = min(box1[3], box2[3]) - max(box1[1], box2[1])

    # 计算交集、并集面积
    inter = 0 if in_w <= 0 or in_h <= 0 else in_h * in_w
    union = (box2[2] - box2[0]) * (box2[3] - box2[1]) + \
            (box1[2] - box1[0]) * (box1[3] - box1[1]) - inter
    # 计算IoU
    iou = inter / union

    # 计算空白面积
    blank_area = area_c - union
    # 计算空白部分占比
    blank_count = blank_area / area_c
    giou = iou - blank_count
    return giou

def calculate_diou(box1, box2):
    # 计算两个图像的最小外接矩形的面积
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    area_c = (max(x2, x4) - min(x1, x3)) * (max(y4, y2) - min(y3, y1))

    # 计算中间矩形的宽高
    in_w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_h = min(box1[3], box2[3]) - max(box1[1], box2[1])

    # 计算交集、并集面积
    inter = 0 if in_w <= 0 or in_h <= 0 else in_h * in_w
    union = (box2[2] - box2[0]) * (box2[3] - box2[1]) + \
            (box1[2] - box1[0]) * (box1[3] - box1[1]) - inter

    # 计算IoU
    iou = inter / union

    # 计算中心点距离的平方
    center_dist = np.square((x1 + x2) / 2 - (x3 + x4) / 2) + \
                  np.square((y1 + y2) / 2 - (y3 + y4) / 2)

    # 计算对角线距离的平方
    diagonal_dist = np.square(max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) + \
                    np.square(max(y1, y2, y3, y4) - min(y1, y2, y3, y4))

    # 计算DIoU
    diou = iou - center_dist / diagonal_dist
    return diou


def tlwh_to_xyxy(box):
    """
    将边界框从 tlwh 格式转换为 xyxy 格式。

    Args:
        box (list or tuple): 边界框，格式为 [x, y, w, h]。

    Returns:
        list: 转换后的边界框，格式为 [x1, y1, x2, y2]。
    """
    x, y, w, h = box
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return [x1, y1, x2, y2]


def linear_assignment(cost_matrix, thresh, use_lap=True):
    """
    Perform linear assignment using scipy or lap.lapjv.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments.
        thresh (float): Threshold for considering an assignment valid.
        use_lap (bool, optional): Whether to use lap.lapjv. Defaults to True.

    Returns:
        (tuple): Tuple containing matched indices, unmatched indices from 'a', and unmatched indices from 'b'.
    """

    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    if use_lap:
        # https://github.com/gatagat/lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        x, y = scipy.optimize.linear_sum_assignment(cost_matrix)  # row x, col y
        matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh])
        if len(matches) == 0:
            unmatched_a = list(np.arange(cost_matrix.shape[0]))
            unmatched_b = list(np.arange(cost_matrix.shape[1]))
        else:
            unmatched_a = list(set(np.arange(cost_matrix.shape[0])) - set(matches[:, 0]))
            unmatched_b = list(set(np.arange(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def diou_distance(atracks, btracks):  # diou
    """
        从 atracks 和 btracks 中提取检测框，将其从 tlwh 格式转换为 xyxy 格式，
        并使用 calculate_diou 函数计算 DIoU 矩阵。

        Args:
            atracks (list[STrack] | list[np.ndarray]): 轨迹 'a' 的列表或边界框列表，格式为 tlwh。
            btracks (list[STrack] | list[np.ndarray]): 轨迹 'b' 的列表或边界框列表，格式为 tlwh。

        Returns:
            np.ndarray: 计算得到的 DIoU 矩阵。
        """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = [tlwh_to_xyxy(box) for box in atracks]
        btlbrs = [tlwh_to_xyxy(box) for box in btracks]
    else:
        atlbrs = [tlwh_to_xyxy(track.tlwh) for track in atracks]
        btlbrs = [tlwh_to_xyxy(track.tlwh) for track in btracks]

    dious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    for i in range(len(atlbrs)):
        for j in range(len(btlbrs)):
            dious[i][j] = calculate_diou(atlbrs[i], btlbrs[j])

    # 计算 1 - iou
    cost_matrix = 1 - dious
    # 如果 1 - iou 小于零，让该值变为 0
    # cost_matrix[cost_matrix < 0] = 0

    return cost_matrix  # cost matrix

def giou_distance(atracks, btracks):  # giou
    """
        从 atracks 和 btracks 中提取检测框，将其从 tlwh 格式转换为 xyxy 格式，
        并使用 calculate_giou 函数计算 GIoU 矩阵。

        Args:
            atracks (list[STrack] | list[np.ndarray]): 轨迹 'a' 的列表或边界框列表，格式为 tlwh。
            btracks (list[STrack] | list[np.ndarray]): 轨迹 'b' 的列表或边界框列表，格式为 tlwh。

        Returns:
            np.ndarray: 计算得到的 DIoU 矩阵。
        """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = [tlwh_to_xyxy(box) for box in atracks]
        btlbrs = [tlwh_to_xyxy(box) for box in btracks]
    else:
        atlbrs = [tlwh_to_xyxy(track.tlwh) for track in atracks]
        btlbrs = [tlwh_to_xyxy(track.tlwh) for track in btracks]

    gious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    for i in range(len(atlbrs)):
        for j in range(len(btlbrs)):
            gious[i][j] = GIoU(atlbrs[i], btlbrs[j])

    # 计算 1 - iou
    cost_matrix = 1 - gious
    # 如果 1 - iou 小于零，让该值变为 0
    # cost_matrix[cost_matrix < 0] = 0

    return cost_matrix  # cost matrix

def tiou_distance(atracks, btracks):  # tiou
    """
    Compute cost based on Intersection over Union (IoU) between tracks.
    Args:
        atracks (list[STrack] | list[np.ndarray]): List of tracks 'a' or bounding boxes.
        btracks (list[STrack] | list[np.ndarray]): List of tracks 'b' or bounding boxes.

    Returns:
        (np.ndarray): Cost matrix computed based on IoU.
    """
    # 用于存储atracks中检测框的对角线长度
    atracks_diagonal_lengths = []
    # 用于存储btracks中检测框的对角线长度
    btracks_diagonal_lengths = []

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) \
            or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        # 遍历atracks，提取每个元素的_tlwh属性并计算tan
        for track in atracks:
            if hasattr(track, '_tlwh'):
                diagonal_length = calculate_diagonal_length(track._tlwh)
                atracks_diagonal_lengths.append(diagonal_length)

        # 遍历btracks，提取每个元素的_tlwh属性并计算tan
        for track in btracks:
            if hasattr(track, '_tlwh'):
                diagonal_length = calculate_diagonal_length(track._tlwh)
                btracks_diagonal_lengths.append(diagonal_length)

        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    # 确定自定义变量，用于调整计算出的IOU
    custom_factors = np.zeros((len(atlbrs), len(btlbrs)))
    for i, a_dl in enumerate(atracks_diagonal_lengths):
        for j, b_dl in enumerate(btracks_diagonal_lengths):
            if a_dl == 0 or b_dl == 0:
                custom_factors[i][j] = 0  # 避免除以 0 的情况，可根据需要调整
            else:
                custom_factors[i][j] = max(a_dl, b_dl) / min(a_dl, b_dl)
                # custom_factors[i][j] = 1 - min(a_dl, b_dl) / max(a_dl, b_dl)

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if len(atlbrs) and len(btlbrs):
        ious = bbox_ioa(np.ascontiguousarray(atlbrs, dtype=np.float32),
                        np.ascontiguousarray(btlbrs, dtype=np.float32),
                        iou=True)

    # 将计算出的IOU乘以对应的自定义变量
    ious *= custom_factors

    # 计算 1 - iou
    cost_matrix = 1 - ious
    # 如果 1 - iou 小于零，让该值变为 0
    # cost_matrix[cost_matrix < 0] = 0

    return cost_matrix  # cost matrix


# def iou_distance_copy(atracks, btracks):
def iou_distance(atracks, btracks):  # iou
    """
    Compute cost based on Intersection over Union (IoU) between tracks.

    Args:
        atracks (list[STrack] | list[np.ndarray]): List of tracks 'a' or bounding boxes.
        btracks (list[STrack] | list[np.ndarray]): List of tracks 'b' or bounding boxes.

    Returns:
        (np.ndarray): Cost matrix computed based on IoU.
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) \
            or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if len(atlbrs) and len(btlbrs):
        ious = bbox_ioa(np.ascontiguousarray(atlbrs, dtype=np.float32),
                        np.ascontiguousarray(btlbrs, dtype=np.float32),
                        iou=True)
    return 1 - ious  # cost matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    Compute distance between tracks and detections based on embeddings.

    Args:
        tracks (list[STrack]): List of tracks.
        detections (list[BaseTrack]): List of detections.
        metric (str, optional): Metric for distance computation. Defaults to 'cosine'.

    Returns:
        (np.ndarray): Cost matrix computed based on embeddings.
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Normalized features
    return cost_matrix


def fuse_score(cost_matrix, detections):
    """
    Fuses cost matrix with detection scores to produce a single similarity matrix.

    Args:
        cost_matrix (np.ndarray): The matrix containing cost values for assignments.
        detections (list[BaseTrack]): List of detections with scores.

    Returns:
        (np.ndarray): Fused similarity matrix.
    """

    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim  # fuse_cost
