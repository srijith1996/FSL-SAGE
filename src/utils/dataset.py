#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.7

import os
import json
import numpy as np
from torch.utils.data import Dataset
  
    
class FEMNIST(Dataset):
    def __init__(self, dataDir, train = False, transform = None):
        super(FEMNIST, self).__init__()
        self.train, self.trainsform = train, transform

        if self.train:
            filePath = os.path.join(dataDir, 'data/train')
        else:
            filePath = os.path.join(dataDir, 'data/test')

        files = os.listdir(filePath)
        files = [f for f in files if f.endswith('.json')]
        files.sort()

        self.imgs, self.labels, self.num = [], [], []

        for f in files:
            jsonPath = os.path.join(filePath, f)
            with open(jsonPath, 'r') as inf:
                data = json.load(inf)

            for ud in data['user_data'].values():
                self.imgs.extend(ud['x'])
                self.labels.extend(ud['y'])
            self.num.extend(data['num_samples'])

        self.imgs = np.asarray(self.imgs).reshape(-1, 28, 28)
        self.labels = np.asarray(self.labels)


    def __getitem__(self, index):
        return (self.imgs[index], self.labels[index])

    def __len__(self):
        return min(len(self.imgs), len(self.labels))