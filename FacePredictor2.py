# -*- coding: utf-8 -*-
"""
# @Date: 2021/2/25 下午9:22
# @Author: yinhao
# @Email: yinhao_x@163.com
# @Github: http://github.com/yinhaoxs
# @Software: PyCharm
# @File: faceInterface.py
# Copyright @ 2020 yinhao. All rights reserved.
"""
import sys
sys.path.append('.')
import torch
import numpy as np
import os
from PIL import Image
import cv2
from .models import net
from .models.DBFace import DBFace
from .models import common
from .utils.detection import DBFaceProcess
import json
import hashlib

def generate_md5(input_string):
    # 创建一个MD5对象
    md5_hash = hashlib.md5()
    # 将字符串编码并更新MD5对象
    md5_hash.update(input_string.encode('utf-8'))
    # 获取MD5哈希值
    md5_value = md5_hash.hexdigest()
    return md5_value

class Dbface2Adaface():
    def __init__(self, device):
        self.root = os.path.dirname(os.path.abspath(__file__))
        # 1 设置模型参数
        self.device = device
        self.fixed_size = 1080
        self.threshold = 0.75
        # 2 加载人脸检测模型
        self.dbfaceprocess = DBFaceProcess()
        self.model_db = DBFace()
        self.model_db.load_state_dict(torch.load(self.root + "/checkpoints/dbface.pth", map_location=self.device))
        self.model_db.to(self.device)
        self.model_db.eval()
        self.mean = [0.408, 0.447, 0.47]
        self.std = [0.289, 0.274, 0.278]
        # 3.加载人脸识别模块
        self.adaface_models = {'ir_101': self.root + "/checkpoints/adaface_ir101_webface12m.ckpt"}
        self.model = net.build_model('ir_101')
        self.statedict = torch.load(self.adaface_models['ir_101'], map_location=self.device)['state_dict']
        self.model_statedict = {key[6:]: val for key, val in self.statedict.items() if key.startswith('model.')}
        self.model.load_state_dict(self.model_statedict)
        self.model.to(self.device)
        self.model.eval()

    # part1:按比例缩放输入图片
    def pretreatment(self, img):
        w, h = img.size
        if max(h, w) < self.fixed_size:
            img = img.copy()
        else:
            if h >= w:
                factor = h / float(self.fixed_size)
                new_w = int(w / factor)
                img = img.resize((new_w, self.fixed_size))
            else:
                factor = w / float(self.fixed_size)
                new_h = int(h / factor)
                img = img.resize((self.fixed_size, new_h))
        return img

    # part2:图片归一化,作为torch的输入
    def to_input(self, pil_rgb_image):
        np_img = np.array(pil_rgb_image)
        brg_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
        tensor = torch.tensor([brg_img.transpose(2, 0, 1)]).float()
        return tensor

    # part3:检测图片中的人脸
    def align(self, img):
        # 人脸检测推理部分
        img_ori = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = common.pad(img_ori)
        img = ((img / 255.0 - self.mean) / self.std).astype(np.float32)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)[None].to(self.device)
        with torch.set_grad_enabled(False):
            hm, box, landmark = self.model_db(img)
        bboxes, faces = self.dbfaceprocess.align_multi(img_ori, hm, box, landmark, threshold=0.4, nms_iou=0.5)
        return faces, bboxes

    # part4:提取图片中的人脸特征
    def userFeatureExtra(self, faces):
        embs = []
        face_Nums = len(faces)
        for face in faces:
            face = self.to_input(face)
            with torch.set_grad_enabled(False):
                emb, _ = self.model(face.to(self.device))
                embs.append(emb)
        embs_user = torch.cat(embs)
        return embs_user

    # part5:写真库中的人脸检测与特征提取,保留单张人脸的特征作为比对
    def photoFeatureExtra(self, singer_id):
        # 根据歌手ID加载对应的json
        embeddings_file = self.root + "/checkpoints/dispersion/{}.json".format(singer_id)

        # 若embedding文件存在,则直接获取对应歌手的特征向量
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'r') as f:
                photo_embedding = json.load(f).get(str(singer_id))
            # 若特征向量不存在,则返回None
            if photo_embedding is None:
                return None
            else:
                return torch.tensor(photo_embedding).to(self.device)
        else:
            return None

    # part6:写真人脸库特征与用户上传人脸特征比对
    def verify(self, embs_user, embs_photo):
        # 扩展embs_user的第一个维度，使其与embs_photo的第一个维度相同
        embs_user_expanded = embs_user.expand(embs_photo.shape[0], -1) 
        # 计算余弦相似度
        cos_dist = torch.cosine_similarity(embs_user_expanded, embs_photo, dim=1)
        cos_dist = cos_dist * 0.5 + 0.5
        # 找到最相似的索引及其相似度值
        most_similar_index = torch.argmax(cos_dist).item()
        most_similar_value = cos_dist[most_similar_index].item()
        # 计算相似度是否超过当前设置的阈值
        if most_similar_value >= self.threshold:
            return True
        else: 
            return False

    # part7:用户上传图片数据加载
    def faceInference(self, singer_id, singer_path):
        try:
            # 1.有无人像(portrait) 2.是否组合、二次元、动漫(group) 3.是否有歌手人像信息(basemap) 4.是否为歌手本人(match)
            results_dict = {}
            results_dict["basemap"] = False
            results_dict["match"] = False
            # 新增**
            img = Image.open(singer_path).convert("RGB")
            # 前处理
            img = self.pretreatment(img)
            # 1.人脸检测
            faces, bboxes = self.align(img)
            if len(faces) == 0:
                results_dict["portrait"] = False
                results_dict["group"] = False
            elif len(faces) > 1:
                results_dict["portrait"] = True
                results_dict["group"] = True
            else:
                results_dict["portrait"] = True
                results_dict["group"] = False
                # 2.用户上传图片的人脸特征提取
                embs_user = self.userFeatureExtra(faces)
                # 3.写真库人脸特征提取
                embs_photo = self.photoFeatureExtra(singer_id)
                if embs_photo is not None:
                    match = self.verify(embs_user, embs_photo)
                    results_dict["basemap"] = True
                    results_dict["match"] = match
                else:
                    results_dict["basemap"] = False
                    results_dict["match"] = False
            return results_dict
        except:
            return None


if __name__ == '__main__':
    # 歌手ID
    singer_id = 300218
    # 用户上传的歌手写真
    singer_path = "/data3/haoyin/videostardetect/images/photo/300218/微信图片_20240508111257.png"
    # 预测
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    Predictor = Dbface2Adaface(device = device)
    face_result = Predictor.faceInference(singer_id, singer_path)
    print("face_result:{}".format(face_result))