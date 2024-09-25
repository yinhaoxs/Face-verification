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
from models import net
from models.DBFace import DBFace
from models import common
from utils.detection import DBFaceProcess
import json


class Dbface2Adaface():
    def __init__(self, device):
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.device = device
        self.fixed_size = 1080
        self.threshold = 0.7
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


    # 0.1 保存嵌入向量:将新的嵌入向量保存到字典中并写入磁盘
    def save_embedding(self, embeddings_file, singer_embeddings, singer_id, embedding):
        if singer_embeddings.get(str(singer_id)) is None:
            singer_embeddings[str(singer_id)] = embedding
        self.save_embeddings_to_disk(embeddings_file, singer_embeddings)

    # 0.2 从字典获取嵌入向量:从字典中获取已保存的嵌入向量
    def get_embedding_from_dict(self, singer_embeddings, singer_id):
        return singer_embeddings.get(singer_id)

    # 0.3 加载嵌入向量字典:从磁盘加载嵌入向量字典
    def load_embeddings_from_disk(self, embeddings_file):
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'r') as f:
                return json.load(f)
        return {}

    # 0.4 保存嵌入向量字典:将嵌入向量字典保存到磁盘     
    def save_embeddings_to_disk(self, embeddings_file, singer_embeddings):
        with open(embeddings_file, 'w') as f:
            json.dump(singer_embeddings, f, indent=4) 

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
    def photoFeatureExtra(self, photo_dir, f_error):
        # 保存没有人脸、有多个人脸的ID
        f = open(f_error, "w")

        # 遍历根文件夹中的每个子文件夹
        num = 0
        for subdir in os.listdir(photo_dir):
            num += 1
            subdir_path = os.path.join(photo_dir, subdir)
            singer_id = subdir_path.split("/")[-1]
            if num % 100 == 0:
                print("歌手数量:{}, 歌手ID:{}".format(num, singer_id))

            # 保存到对应的json文件
            embeddings_file = self.root + "/checkpoints/dispersion/{}.json".format(singer_id)
            singer_embeddings = self.load_embeddings_from_disk(embeddings_file)

            # 检查是否是一个文件夹
            if os.path.isdir(subdir_path):
                embs_photo = []

                # 遍历子文件夹中的每个文件
                for file in os.listdir(subdir_path):
                    try:
                        file_path = os.path.join(subdir_path, file)
                        # 复原
                        img = Image.open(file_path).convert("RGB")
                        img = self.pretreatment(img)
                        faces, bboxes = self.align(img)
                        if len(faces) == 0:
                            continue
                        elif len(faces) == 1:
                            embedding = self.userFeatureExtra(faces)
                            embs_photo.append(embedding)
                        else:
                            continue
                    except:
                        f.write(str(file_path) + "|" + str(singer_id) + "|" + "0" + "\n")
            if len(embs_photo) == 0:
                # print("error:{}".format(singer_id))
                f.write(str(file_path) + "|" + str(singer_id) + "|" + "1" + "\n")
            else:
                # 保存为张量
                embs_photo = torch.cat(embs_photo)
                # 保存歌手id与特征到json字典
                self.save_embedding(embeddings_file, singer_embeddings, str(singer_id), embs_photo.tolist())
        f.close()

    # part7:用户上传图片数据加载
    def faceInference(self, photo_dir, f_error):
        try:
            # 写真库人脸特征提取
            embs_photo = self.photoFeatureExtra(photo_dir, f_error)
        except:
            print("error")


if __name__ == '__main__':
    # 错误日志保存
    f_error = "/data3/haoyin/videostardetect/dispersion_error.txt"
    # 歌手写真素材目录
    photo_dir = "/data3/haoyin/videostardetect/photo_images/"
    # 预测
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    Predictor = Dbface2Adaface(device = device)
    face_result = Predictor.faceInference(photo_dir, f_error)
    print("face_result:{}".format(face_result))