import time
import torch
import os, numpy as np
import os.path as osp
import shutil
import cv2
from tqdm import tqdm
import time


def generate_edge(label_dir, edge_dir):
    """Generate edges for labels in label_dir and save them to edge_dir
    """
    print('Generating edges from {} to {}'.format(label_dir, edge_dir))
    # if edge_dir exist -> delete folder and all file.
    if os.path.exists(edge_dir):
        shutil.rmtree(edge_dir)
    # make cean edge_dir
    os.makedirs(edge_dir)
    # get list of label file in label_dir
    ll = os.listdir(label_dir)
    # 
    for idx, filename in tqdm(enumerate(ll)):
        if idx == 0 or idx % 1000:
            print(filename)
        label = cv2.imread(osp.join(label_dir, filename), cv2.IMREAD_GRAYSCALE)
        edge = np.zeros_like(label)
        # label shape = (H,W) ex) (512,512)
        # check all pixel in label image
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                flag = 1
                # check left,right,up,down pixel
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    # i,j is current pixel, and x, y is around pixel
                    x = i + dx
                    y = j + dy
                    # range is 0 to H or W
                    if 0 <= x < label.shape[0] and 0 <= y < label.shape[1]:
                        # if pixel(i,j) and (x,y) is different
                        # then change pixel(i,j) to edge(255)
                        if label[i,j] != label[x,y]:
                            edge[i,j] = 255
        cv2.imwrite(osp.join(edge_dir, filename), edge)

# train
# generate_edge('NIA_train_dataset/labels/', 'NIA_train_dataset/edges/')

# inference for real data
generate_edge('test_label_nia/', 'inference/edges/')

# test
# generate_edge('NIA_train_dataset/test_label/', 'NIA_train_dataset/test_edges/')
