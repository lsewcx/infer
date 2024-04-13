import cv2
from typing import List
import numpy as np

'''
本文件主要为了实现各种nms算法
'''

def cv2nms(boxes:List,scores:int,score_threshold:int, nms_threshold:int)->List:
    '''
    使用opencv的nms算法
    参数:
    - boxes: 边界框坐标列表，每个元素是一个列表，包含边界框的坐标[x_min, y_min, x_max, y_max]。
    - scores: 边界框的置信度列表，每个元素是一个浮点数。
    - score_threshold: 置信度阈值，低于该阈值的边界框将被过滤。
    - nms_threshold: 非最大抑制的阈值，用于过滤重叠度高的边界框。
    返回:
    - indices: 保留的边界框索引列表，每个元素是一个整数。
    '''
    return cv2.dnn.NMSBoxes(boxes, scores, score_threshold=score_threshold, nms_threshold=nms_threshold)

def nms(boxes:List,scores:int,score_threshold:int, nms_threshold:int)->List:
    '''
    使用自己实现的nms算法,主要用来学习原理
    参数:
    - boxes: 边界框坐标列表，每个元素是一个列表，包含边界框的坐标[x_min, y_min, x_max, y_max]。
    - scores: 边界框的置信度列表，每个元素是一个浮点数。
    - score_threshold: 置信度阈值，低于该阈值的边界框将被过滤。
    - nms_threshold: 非最大抑制的阈值，用于过滤重叠度高的边界框。
    返回:
    - keep: 保留的边界框索引列表，每个元素是一个整数。
    '''
    indices = np.where(np.array(scores) >= score_threshold)[0]
    boxes = np.array(boxes)[indices]
    scores = np.array(scores)[indices]

    # 对边界框按照置信度得分排序
    order = scores.argsort()[::-1]

    keep = []  # 用于保存最终保留的边界框的索引
    while order.size > 0:
        i = order[0]  # 取出当前最大的置信度得分的边界框
        keep.append(indices[i])

        # 计算当前边界框与其他所有边界框的交并比
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / ((boxes[i, 2] - boxes[i, 0] + 1) * (boxes[i, 3] - boxes[i, 1] + 1) +
                       (boxes[order[1:], 2] - boxes[order[1:], 0] + 1) *
                       (boxes[order[1:], 3] - boxes[order[1:], 1] + 1) - inter)

        # 保留交并比小于阈值的边界框
        inds = np.where(ovr <= nms_threshold)[0]
        order = order[inds + 1]

    return keep
