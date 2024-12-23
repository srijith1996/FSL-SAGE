#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.7
import numpy as np

def sample_iid(num_clients, dataset):
    '''
        Sample i.i.d client data from the givend dataset;

        Return:
            The sample indexs and number of each client (dict.)
    '''

    ## sample number for each client (balanced)
    itemNum = int(len(dataset)/num_clients)

    if itemNum <= 0:
        exit(f"[ERROR] Data is not enough to depart to {num_clients} clients.")

    allIdx = [i for i in range(len(dataset))]
    dictUser = {i: {'idxs': np.array([])} for i in range(num_clients)}

    ## set random seed
    #np.random.seed(15688)

    for i in range(num_clients):
        dictUser[i]['num'] = itemNum
#         np.random.seed(2000)
        dictUser[i]['idxs'] = set(np.random.choice(allIdx, itemNum, replace=False))
        ## remove the sample indexs been distributed
        allIdx = list(set(allIdx) - dictUser[i]['idxs'])

    return dictUser


def sample_noniid(num_clients, dataset_name, dataset, shard_num):
    '''
        Sample Non-i.i.d client data from the givend dataset;

        Return:
            The sample indexs and number of each client (dict.)
    '''
    if dataset_name in ['shakespeare', 'femnist']: # non-iid and unbalanced
        num = len(dataset.num)      
        dictUser = {i: {'idxs': np.array([])} for i in range(num_clients)}
        firstIdx = 0

        if num_clients > num:
            exit(f"[ERROR] Client amount should be less than {num}.")
        else:
            itemNum = int(len(dataset)/num_clients)
            for i in range(num_clients):
                dictUser[i]['num'] = itemNum
                dictUser[i]['idxs']= [(firstIdx + i) for i in range(itemNum)]
                firstIdx += itemNum

    else:
        itemNum = int(len(dataset)/num_clients)
        if itemNum <= 0:
            exit(f"[ERROR] Data is not enough to depart to {num_clients} clients.")

        ## the number of shards and the number of samples in each shard
        shardNum = shard_num * num_clients
        imageNum = int(len(dataset) / shardNum)

        shardIdxs = [i for i in range(shardNum)]
        dictUser = {i: {'idxs': np.array([])} for i in range(num_clients)}

        ## sort the samples according to label
        idx = np.arange(shardNum * imageNum)
        label = np.array(dataset.targets)[:len(idx)]
        idx_label = np.vstack((idx, label))
        idx_label = idx_label[:, idx_label[1, :].argsort()]
        idx = idx_label[0, :]

        ## divide and assign 'shard_num' shards for each client
        for i in range(num_clients):
            np.random.seed(16511)
#             np.random.seed(2000)
            randSet = set(np.random.choice(shardIdxs, shard_num, replace = False))
            shardIdxs = list(set(shardIdxs) - randSet)

            for rand in randSet:
                dictUser[i]['idxs'] = np.concatenate((dictUser[i]['idxs'],
                                idx[rand*imageNum: (rand+1)*imageNum]),axis=0)

            dictUser[i]['num'] = len(randSet) * imageNum

    return dictUser

def noniid_dirichlet_equal_split(num_clients, dataset, alpha, num_classes):
    """Construct a federated dataset from the centralized CIFAR-10.
    Sampling based on Dirichlet distribution over categories, following the paper
    Measuring the Effects of Non-Identical Data Distribution for
    Federated Visual Classification (https://arxiv.org/abs/1909.06335).
    Args:
        dataset: The dataset to split
        alpha: Parameter of Dirichlet distribution. Each client
        samples from this Dirichlet to get a multinomial distribution over
        classes. It controls the data heterogeneity of clients. If approaches 0,
        then each client only have data from a single category label. If
        approaches infinity, then the client distribution will approach IID
        partitioning.
        num_clients: The number of clients the examples are going to be partitioned on.
        num_classes: The number of unique classes in the dataset
    Returns:
        a dict where keys are client numbers from 0 to num_clients and nested dict inside of each key has keys train
        and validation containing arrays of the indicies of each sample.
        """
    labels = np.array(dataset.targets)
    dict_users = {}
    multinomial_vals = []
    examples_per_label = []
    for i in range(num_classes):
        examples_per_label.append(int(np.argwhere(labels == i).shape[0]))

    # Each client has a multinomial distribution over classes drawn from a Dirichlet.
    for i in range(num_clients):
        proportion = np.random.dirichlet(alpha * np.ones(num_classes))
        multinomial_vals.append(proportion)

    multinomial_vals = np.array(multinomial_vals)
    example_indices = []

    for k in range(num_classes):
        label_k = np.where(labels == k)[0]
        np.random.shuffle(label_k)
        example_indices.append(label_k)

    example_indices = np.array(example_indices, dtype=object)

    client_samples = [[] for _ in range(num_clients)]
    count = np.zeros(num_classes).astype(int)

    examples_per_client = int(labels.shape[0] / num_clients)

    for k in range(num_clients):
        for i in range(examples_per_client):
            sampled_label = np.argwhere(np.random.multinomial(1, multinomial_vals[k, :]) == 1)[0][0]
            label_indices = example_indices[sampled_label]
            client_samples[k].append(label_indices[count[sampled_label]])
            count[sampled_label] += 1
            if count[sampled_label] == examples_per_label[sampled_label]:
                multinomial_vals[:, sampled_label] = 0
                multinomial_vals = (
                        multinomial_vals /
                        multinomial_vals.sum(axis=1)[:, None])
    for i in range(num_clients):
        np.random.shuffle(np.array(client_samples[i]))
        samples = np.array(client_samples[i])
        train_idxs = samples.astype('int64').squeeze()
        #validation_idxs = samples[int(samples.shape[0] * 0.9):].astype('int64').squeeze()

        dict_users[i] = {}
        dict_users[i]['idxs'] = train_idxs
        dict_users[i]['num'] = len(train_idxs)
        #dict_users[i]['validation'] = validation_idxs

    return dict_users