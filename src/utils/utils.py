#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.7
import random
from trains import sample
from utils import dataset
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def get_dataset(args, u_args):
    '''
        Download the dataset (if needed) and depart it for clients.

        Returns:
            trainSet:	The whole training set
            testSet:	The whole test set
            userGroup:	The sample indexs of each client (dict.)
    '''

    if args['dataset'] == 'cifar':
        dataDir = '../datas/cifar'
        
        random.seed(10)
        ## define the image transform rule
        trainRule = transforms.Compose([
            transforms.RandomCrop(24),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.5, contrast=(0.2,1.8)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])
        testRule = transforms.Compose([
            transforms.CenterCrop(24),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

        ## access to the dataset
        trainSet = datasets.CIFAR10(dataDir, train=True, download=True,
                                    transform=trainRule)
        testSet = datasets.CIFAR10(dataDir, train=False, download=True,
                                    transform=testRule)

    elif args['dataset'] == 'femnist':
        dataDir = '/notebooks/femnist/'

        ## access to the dataset
        trainSet = dataset.FEMNIST(dataDir, train = True)  #-----todo
        testSet  = dataset.FEMNIST(dataDir, train = False)
        print(trainSet.imgs.shape)
        print(testSet.imgs.shape)

    else:
        exit(f"[ERROR] Unrecognized dataset '{args['dataset']}'.")

    return trainSet, testSet


def depart_dataset(args, s_args, trainSet, testSet):
    '''
        Depart the whole dataset for clients.

        Return:
            clientTrainSets: a dict. of training data idxs keyed by client number.
            clientTestSets: a dict. of test data idxs keyed by client number.
    '''
    if args['sample'] == 'iid':
        clientTrainSets = sample.sample_iid(args, trainSet)  #------todo
        clientTestSets = sample.sample_iid(args, testSet)
    elif args['sample'] == 'noniid':
        clientTrainSets = sample.sample_noniid(args, s_args, trainSet)
        clientTestSets = sample.sample_noniid(args, s_args, testSet)
    else:
        exit(f"[ERROR] Illegal sample method '{args['sample']}.")

    return (clientTrainSets, clientTestSets)


class DatasetSplit(Dataset):
    '''
        An abstract Dataset class wrapped around Pytorch Dataset class
    '''
    def __init__(self, dataset, idx):
        self.dataset = dataset
        self.idx = [int(i) for i in idx]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        inputs, labels = self.dataset[self.idx[item]]
        return inputs, labels


def show_utils(args):
    '''
        Print system setup profile.
    '''

    print('*'*80); print('SYSTEM CONFIGS'); print('*'*80)
    print(f"  \\__ Method:        {args['method']}")
    print(f"  \\__ Dataset:       {args['dataset']}")
    print(f"  \\__ Save:          {args['save']}")

    if args['method'] == 'CSE_FSL':
        print(f"  \\__ Sample method: {args['sample']}")
