import numpy as np


def biou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
             + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)

    x1 = bboxes1[..., 2] - bboxes1[..., 0]
    y1 = bboxes1[..., 3] - bboxes1[..., 1]
    # bboxes1_angle_tan = x1 / (y1 + + 1e-6)
    bboxes1_angle = np.arctan2(y1, x1)

    x2 = bboxes2[..., 2] - bboxes2[..., 0]
    y2 = bboxes2[..., 3] - bboxes2[..., 1]
    bboxes2_angle = np.arctan2(y2, x2)

    angle_diff = np.minimum(bboxes1_angle, bboxes2_angle) / np.maximum(bboxes1_angle, bboxes2_angle)
    biou = angle_diff * o
    return biou