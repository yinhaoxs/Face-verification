# -*- coding: utf-8 -*-
"""
# @Date: 2021/4/12 下午5:50
# @Author: yinhao
# @Email: yinhao_x@163.com
# @Github: http://github.com/yinhaoxs
# @Software: PyCharm
# @File: detection.py
# Copyright @ 2020 yinhao. All rights reserved.
"""
try:
    from ..models import common
    from ..models.common import intv
    from ..utils.align_trans import get_reference_facial_points, warp_and_crop_face
except:
    from models import common
    from models.common import intv
    from utils.align_trans import get_reference_facial_points, warp_and_crop_face

import numpy as np
import torch.nn.functional as F
from PIL import Image


class DBFaceProcess():
    def __init__(self):
        # self.device = device
        pass

    def nms(self, objs, iou=0.5):

        if objs is None or len(objs) <= 1:
            return objs

        objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
        keep = []
        flags = [0] * len(objs)
        for index, obj in enumerate(objs):

            if flags[index] != 0:
                continue

            keep.append(obj)
            for j in range(index + 1, len(objs)):
                if flags[j] == 0 and obj.iou(objs[j]) > iou:
                    flags[j] = 1
        return keep

    def detect(self, hm, box, landmark, threshold, nms_iou):
        hm_pool = F.max_pool2d(hm, 3, 1, 1)
        scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
        hm_height, hm_width = hm.shape[2:]

        scores = scores.squeeze()
        indices = indices.squeeze()
        ys = list((indices / hm_width).int().data.numpy())
        xs = list((indices % hm_width).int().data.numpy())
        scores = list(scores.data.numpy())
        box = box.cpu().squeeze().data.numpy()
        landmark = landmark.cpu().squeeze().data.numpy()

        stride = 4
        objs = []
        for cx, cy, score in zip(xs, ys, scores):
            if score < threshold:
                break

            x, y, r, b = box[:, cy, cx]
            xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
            x5y5 = landmark[:, cy, cx]
            x5y5 = (common.exp(x5y5 * 4) + ([cx] * 5 + [cy] * 5)) * stride
            box_landmark = list(zip(x5y5[:5], x5y5[5:]))
            objs.append(common.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))
        return self.nms(objs, iou=nms_iou)

    def detect_faces(self, hm, box, landmark, threshold, nms_iou):
        # 检测人脸与关键点
        bounding_boxes, landmarks = [], []
        objs = self.detect(hm, box, landmark, threshold, nms_iou)
        for obj in objs:
            x, y, r, b = intv(obj.box)
            bounding_boxes.append([float(x), float(y), float(r), float(b)])#人脸box的位置
            # 获取dbface模型的landmark点
            if obj.haslandmark:
                for i in range(len(obj.landmark)):
                    x, y = obj.landmark[i][:2]
                    landmarks.append([float(x), float(y)])

        return np.array(bounding_boxes), np.array(landmarks)

    # 修改代码
    def align_multi(self, img_ori, hm, box, landmark, threshold, nms_iou):
        # 检测人脸
        bboxes, landmarks = self.detect_faces(hm, box, landmark, threshold, nms_iou)
        faces = []
        for i in range(0, len(landmarks) // 5):
            facial5points = [landmarks[i * 5 + j] for j in range(5)]
            # print(f"facial5points:{facial5points}")
            warped_face = warp_and_crop_face(np.array(img_ori), facial5points, get_reference_facial_points(default_square=True), crop_size=(112, 112))
            faces.append(Image.fromarray(warped_face))
        return bboxes, faces

