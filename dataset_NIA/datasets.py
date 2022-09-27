import os
import numpy as np
import random
import torch
import cv2
import json
from torch.utils import data
import torch.distributed as dist
from utils.transforms import get_affine_transform
import os.path as osp
import math
# import face_alignment
import matplotlib.pyplot as plt

def get_ext(path):
    k = os.listdir(path)
    return k[0][-3:]

def rotationMatrixToEulerAngles(R) :
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

class CelebAMaskHQDataSet(data.Dataset):
    def __init__(self, root, dataset, crop_size=[473, 473], 
    scale_factor=0, rotation_factor=0, ignore_label=255, transform=None, num_idx=566):
    # scale_factor=0.25, rotation_factor=30, ignore_label=255, transform=None):
        """
        :rtype:
        """
        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)
        self.ignore_label = ignore_label
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = 0.5
        self.flip_pairs = [[4, 5], [6, 7]]
        self.transform = transform
        self.dataset = dataset

        list_path = os.path.join(self.root, self.dataset  + '_id.txt')

        self.im_list = [i_id.strip() for i_id in open(list_path)]

        if self.dataset == 'test' and len(self.im_list) < num_idx:
            self.im_list = self.im_list[:num_idx]

        self.number_samples = len(self.im_list)

    def __len__(self):
        return self.number_samples

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

        return center, scale

    def __getitem__(self, index):
        # Load training image

        im_name = self.im_list[index]
        # ㅇㅣㄹㅂㅜㅁㅏㄴ load

        #print(im_name)
        #print(os.path.join(self.root,self.dataset,'images',im_name+'.png'))
        
        # print('root', self.root)

        # 확장자
        im_ext = get_ext(os.path.join(self.root, self.dataset, 'images'))
        label_ext = get_ext(os.path.join(self.root, self.dataset, 'labels'))
        edge_ext = get_ext(os.path.join(self.root, self.dataset, 'edges'))
        # print(self.root)
        grayscale_root_img_list= ['./nia_8_inference/test_2000',  './nia_8_inference/test_init_data', './NIA_8_full/', './nia_8_inference/', './dataset_NIA/', './inference_NIA/',
         './dataset_600_100/', './NIA_8/']
        if self.root in grayscale_root_img_list:
            # print('self.root@@@@@@@@@@@@@@@@')
            label_ext = 'grayscale.' + label_ext
            edge_ext = 'grayscale.' + edge_ext

        # NIA는 전부다 .png인데 CelebA는 원본images가jpg다.
        im_path = os.path.join(self.root, self.dataset, 'images', im_name + '.' + im_ext)
        parsing_anno_path = os.path.join(self.root, self.dataset, 'labels', im_name + '.' + label_ext)
        edge_path = os.path.join(self.root, self.dataset, 'edges', im_name + '.' + edge_ext)
        # print(edge_path)
        # print(im_path,parsing_anno_path,edge_path)
        # print(os.path.isfile(im_path),os.path.isfile(parsing_anno_path),os.path.isfile(edge_path))
        
        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im,(512,512),interpolation=cv2.INTER_AREA)
        edge = cv2.resize(edge,(512,512),interpolation=cv2.INTER_AREA)
        h, w, _ = im.shape
        parsing_anno = np.zeros((h, w), dtype=np.long)

        # Get center and scale
        center, s = self._box2cs([0, 0, w - 1, h - 1])
        # print('center, s ', center, s )
        r = 0

        if self.dataset != 'test' : #or self.dataset != 'test_celebA': 
            # parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)
            parsing_anno = cv2.imread(parsing_anno_path, -1)
            parsing_anno = parsing_anno.astype(np.uint8)
            parsing_anno = cv2.resize(parsing_anno,(512,512),interpolation=cv2.INTER_AREA)

            if self.dataset in 'train':

                sf = self.scale_factor
                rf = self.rotation_factor
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                    if random.random() <= 0.6 else 0

                # if random.random() <= self.flip_prob:
                #     im = im[:, ::-1, :]
                #     parsing_anno = parsing_anno[:, ::-1]

                #     center[0] = im.shape[1] - center[0] - 1
                #     right_idx = [5, 7, 9]
                #     left_idx = [4, 6, 8]
                #     for i in range(0, len(left_idx)):
                #         right_pos = np.where(parsing_anno == right_idx[i])
                #         left_pos = np.where(parsing_anno == left_idx[i])
                #         parsing_anno[right_pos[0], right_pos[1]] = left_idx[i]
                #         parsing_anno[left_pos[0], left_pos[1]] = right_idx[i]
        # factor 변화 값 확인 , 현재는 따로 transforom 없음.
        # print('transform factor : ', center, s, r)
        trans = get_affine_transform(center, s, r, self.crop_size)
        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        edge = cv2.warpAffine(
            edge,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        edge[np.where(edge != 0)] = 1

        if self.transform:
            input = self.transform(input)

        meta = {
            'name': im_name,
            'center': center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        # train set output과 test set output 다른데
        # test set은 label_parsing
        if self.dataset not in 'train':
            return input, edge, meta
        else:
            label_parsing = cv2.warpAffine(
                parsing_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255))

            label_parsing = torch.from_numpy(label_parsing)

            return input, label_parsing, edge, meta
