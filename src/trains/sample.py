#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.7
import numpy as np

def sample_iid(args, dataset):
    '''
        Sample i.i.d client data from the givend dataset;

        Return:
            The sample indexs and number of each client (dict.)
    '''

    ## sample number for each client (balanced)
    itemNum = int(len(dataset)/args['client'])

    if itemNum <= 0:
        exit(f"[ERROR] Data is not enough to depart to {args['client']} clients.")

    allIdx = [i for i in range(len(dataset))]
    dictUser = {i: {'idxs': np.array([])} for i in range(args['client'])}

    for i in range(args['client']):
        dictUser[i]['num'] = itemNum
        ## set random seed
        np.random.seed(15688)
#         np.random.seed(2000)
        dictUser[i]['idxs'] = set(np.random.choice(allIdx, itemNum, replace=False))
        ## remove the sample indexs been distributed
        allIdx = list(set(allIdx) - dictUser[i]['idxs'])

    return dictUser


def sample_noniid(args, s_args, dataset):
    '''
        Sample Non-i.i.d client data from the givend dataset;

        Return:
            The sample indexs and number of each client (dict.)
    '''
    if args['dataset'] in ['shakespeare', 'femnist']: # non-iid and unbalanced
        num = len(dataset.num)      
        dictUser = {i: {'idxs': np.array([])} for i in range(args['client'])}
        firstIdx = 0

        if args['client'] > num:
            exit(f"[ERROR] Client amount should be less than {num}.")
        else:
            itemNum = int(len(dataset)/args['client'])
            for i in range(args['client']):
                dictUser[i]['num'] = itemNum
                dictUser[i]['idxs']= [(firstIdx + i) for i in range(itemNum)]
                firstIdx += itemNum

    else:
        itemNum = int(len(dataset)/args['client'])
        if itemNum <= 0:
            exit(f"[ERROR] Data is not enough to depart to {args['client']} clients.")

        ## the number of shards and the number of samples in each shard
        shardNum = args['shard_num'] * args['client']
        imageNum = int(len(dataset) / shardNum)

        shardIdxs = [i for i in range(shardNum)]
        dictUser = {i: {'idxs': np.array([])} for i in range(args['client'])}

        ## sort the samples according to label
        idx = np.arange(shardNum * imageNum)
        label = np.array(dataset.targets)[:len(idx)]
        idx_label = np.vstack((idx, label))
        idx_label = idx_label[:, idx_label[1, :].argsort()]
        idx = idx_label[0, :]

        ## divide and assign 'shard_num' shards for each client
        for i in range(args['client']):
            np.random.seed(16511)
#             np.random.seed(2000)
            randSet = set(np.random.choice(shardIdxs, args['shard_num'],
                                           replace = False))
            shardIdxs = list(set(shardIdxs) - randSet)

            for rand in randSet:
                dictUser[i]['idxs'] = np.concatenate((dictUser[i]['idxs'],
                                idx[rand*imageNum: (rand+1)*imageNum]),axis=0)
            dictUser[i]['num'] = len(randSet)

    return dictUser
