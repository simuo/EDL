import torch
import torchvision
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import cfg
import math
import cv2

img_base_dir = r'D:\VIDI\ExploreDLInIndustry\Betel_nut\imgresize'
img_file_dir = r'label.txt'
transform = torchvision.transforms.ToTensor()


class Mydataset(Dataset):

    def __init__(self):
        with open(img_file_dir) as f:
            self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        labels = {}

        line = self.dataset[index]
        strs = line.split()
        # imgdata = Image.open(os.path.join(img_base_dir, strs[0]))
        im = cv2.imread(img_base_dir + '\\' + strs[0].replace('bmp', 'jpg'), 1)
        imgdata = Image.fromarray(im)
        img = transform(imgdata)

        _boxes = np.array([float(x) for x in strs[1:]])
        boxes = np.split(_boxes, len(_boxes) // 5)

        for feature_size, anchors in cfg.ANCHORS_GROUP.items():
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 6), dtype=np.float32)
            # print(labels[feature_size].shape)

            for box in boxes:
                cls, cx, cy, w, h = box
                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMG_WIDTH)

                for i, anchor in enumerate(anchors):
                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]
                    p_w, p_h = w / anchor[0], h / anchor[1]
                    box_area = w * h

                    # 计算置信度(同心框的IOU(交并))
                    inter = np.minimum(w, anchor[0]) * np.minimum(h, anchor[1])
                    conf = inter / (box_area + anchor_area - inter)

                    labels[feature_size][int(cx_index), int(cy_index), i] = \
                        np.array([cx_offset, cy_offset, np.log(p_w), np.log(p_h), conf, int(cls)])
        return labels[13], labels[26], labels[52], img


if __name__ == '__main__':
    data = Mydataset()
    print(data[0])
