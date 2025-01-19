#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.7
from torchvision.datasets import MNIST, utils
from PIL import Image
import os
import h5py
import torch
import numpy as np
import shutil

def _check_create_train_test(all_examples_file, test_split, train):
    train_file_path = os.path.join(os.path.dirname(all_examples_file), 'train.h5')
    test_file_path = os.path.join(os.path.dirname(all_examples_file), 'test.h5')

    if not (os.path.exists(train_file_path) and os.path.exists(test_file_path)):
        all_examples = h5py.File(all_examples_file, 'r')

        train_images = []
        train_labels = []
        test_images = []
        test_labels = []

        for writer_examples in all_examples.values():
            writer_images = writer_examples['images'][:]
            writer_labels = writer_examples['labels'][:]
            ids = list(range(len(writer_labels)))
            np.random.shuffle(ids)

            test_ids = ids[:int(test_split * len(ids))]
            train_ids = ids[int(test_split * len(ids)):]

            writer_train_images = writer_images[train_ids]
            writer_train_labels = writer_labels[train_ids]
            writer_test_images = writer_images[test_ids]
            writer_test_labels = writer_labels[test_ids]

            train_images.append(writer_train_images)
            train_labels.append(writer_train_labels)
            test_images.append(writer_test_images)
            test_labels.append(writer_test_labels)

        train_images = np.concatenate(train_images, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        test_images = np.concatenate(test_images, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        train_images = np.expand_dims(train_images, axis=1)
        test_images = np.expand_dims(test_images, axis=1)

        train_h5 = h5py.File(train_file_path, 'w')
        train_h5.create_dataset("images", data=train_images, dtype="uint8")
        train_h5.create_dataset("labels", data=train_labels, dtype="uint8")
        train_h5.close()

        test_h5 = h5py.File(test_file_path, 'w')
        test_h5.create_dataset("images", data=test_images, dtype="uint8")
        test_h5.create_dataset("labels", data=test_labels, dtype="uint8")
        test_h5.close()

        all_examples.close()

    data = h5py.File(train_file_path, 'r') if train \
        else h5py.File(test_file_path, 'r')

    return data

class Femnist(torch.utils.data.Dataset):
    def __init__(self, data_dir, train=True):
        self.file_name = os.path.join(data_dir, 'femnist/examples_by_writer.h5')
        self.data = _check_create_train_test(self.file_name, 0.1, train)

    def __len__(self):
        return len(self.data['labels'][:])

    def __getitem__(self, idx):
        img = self.data['images'][idx]
        label = self.data['labels'][idx]
        return torch.Tensor(img), label
