# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 


from __future__ import division
import collections
import numpy as np
import glob
import os
import os.path as osp
from numpy.linalg import norm
from insightface.model_zoo import model_zoo
from insightface_func.utils import face_align

__all__ = ['Face_detect_crop', 'Face']

Face = collections.namedtuple('Face', [
    'bbox', 'kps', 'det_score', 'embedding', 'gender', 'age',
    'embedding_norm', 'normed_embedding',
    'landmark'
])

Face.__new__.__defaults__ = (None, ) * len(Face._fields)


class Face_detect_crop:
    def __init__(self, name, root='~/.insightface_func/models'):
        self.models = {}
        root = os.path.expanduser(root)
        onnx_files = glob.glob(osp.join(root, name, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            if onnx_file.find('_selfgen_')>0:
                #print('ignore:', onnx_file)
                continue
            model = model_zoo.get_model(onnx_file)
            if model.taskname not in self.models:
                print('find model:', onnx_file, model.taskname)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size)
            else:
                model.prepare(ctx_id)

    def get(self, img, crop_size, max_num=0, mode = 'None'):
        bboxes, kpss = self.det_model.detect(img,
                                             threshold=self.det_thresh,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        if mode == 'Both':
            for i in range(bboxes.shape[0]):
                kps = None
                if kpss is not None:
                    kps = kpss[i]
                aimg_None,aimg_arface = face_align.norm_crop(img, kps,crop_size,mode =mode)
            return [aimg_None,aimg_arface]

        else:
            for i in range(bboxes.shape[0]):
                kps = None
                if kpss is not None:
                    kps = kpss[i]
                aimg = face_align.norm_crop(img, kps,crop_size,mode =mode)
            return [aimg]

    def draw_on(self, img, faces):
        import cv2
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(np.int)
            color = (0, 0, 255)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(np.int)
                #print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(img, (kps[l][0], kps[l][1]), 1, color,
                               2)
        return img

