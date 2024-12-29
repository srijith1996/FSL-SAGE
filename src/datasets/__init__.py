# -----------------------------------------------------------------------------
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
from datasets import sample, femnist, nlg_ft

# -----------------------------------------------------------------------------
def get_dataset(cfg, batch_size=None):
    '''
        Download the dataset (if needed) and depart it for clients.

        Returns:
            trainSet:	The whole training set
            testSet:	The whole test set
            userGroup:	The sample indexs of each client (dict.)
    '''

    if 'imagenet' in cfg.name:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    # load configured dataset
    if cfg.name == 'cifar10':
        dataDir = '../datas/cifar'

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
        #random.seed(10)
        ## define the image transform rule
        trainRule = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
            ])
        testRule = transforms.Compose([
            transforms.ToTensor(), 
            normalize
            ])

        ## access to the dataset
        trainSet = datasets.CIFAR10(dataDir, train=True, download=True,
                                    transform=trainRule)
        testSet = datasets.CIFAR10(dataDir, train=False, download=True,
                                    transform=testRule)

    elif cfg.name == 'femnist':
        dataDir = '../datas/femnist'
        trainSet = femnist.Femnist(dataDir, train=True)  #-----todo
        testSet  = femnist.Femnist(dataDir, train=False)

    elif cfg.name == 'cifar100':
        dataDir = '../datas/cifar100'

        train_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform =  transforms.Compose([
                    transforms.Resize(32),
                    transforms.CenterCrop(32),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        trainSet = datasets.CIFAR100(
            dataDir, train=True, download=True, transform=train_transform
        )
        testSet = datasets.CIFAR100(
            dataDir, train=False, download=True, transform=test_transform
        )

    elif cfg.name == 'tinyimagenet':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        data_dir = '../datas/tinyimagenet/tiny-imagenet-200'
        trainSet = datasets.ImageFolder(
            os.path.join(data_dir, 'train'), train_transform
        )
        testSet = datasets.ImageFolder(
            os.path.join(data_dir, 'test'), test_transform
        )

    elif cfg.name == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        dataDir = '../datas/imagenet'
        trainSet = datasets.ImageFolder(
            os.path.join(dataDir, 'train'), transform=train_transform
        )
        testSet = datasets.ImageFolder(
            os.path.join(dataDir, 'val'), transform=test_transform
        )

    elif cfg.name == "nlg_ft":
        dataDir = '../datas/text_completion'

        trainSet = nlg_ft.FT_Dataset(
            f'{dataDir}/data/e2e/train.jsonl', batch_size, cfg.seq_len,
            joint_lm=(cfg.obj == 'jlm')
        )
        testSet  = nlg_ft.FT_Dataset(
            f'{dataDir}/data/e2e/test.jsonl', batch_size, cfg.seq_len,
            joint_lm=(cfg.obj == 'jlm')
        )

    else:
        exit(f"[ERROR] Unrecognized dataset '{cfg.name}'.")

    return trainSet, testSet

# -----------------------------------------------------------------------------
def depart_dataset(num_clients, trainSet, testSet, cfg):
    '''
        Depart the whole dataset for clients.

        Return:
            clientTrainSets: a dict. of training data idxs keyed by client number.
            clientTestSets: a dict. of test data idxs keyed by client number.
    '''
    if cfg.distribution == 'iid':
        clientTrainSets = sample.sample_iid(num_clients, trainSet)  #------todo
        clientTestSets = sample.sample_iid(num_clients, testSet)
    elif cfg.distribution == 'noniid':
        clientTrainSets = sample.sample_noniid(
            num_clients, cfg.name, trainSet, cfg.shard_num
        )
        clientTestSets = sample.sample_noniid(
            num_clients, cfg.name, testSet, cfg.shard_num
        )
    elif cfg.distribution == 'noniid_dirichlet':
        assert 'alpha' in cfg, \
        "config `alpha` required if distribution in `noniid_dirichlet`."
        assert 'num_classes' in cfg, \
        "config `num_classes` required if distribution in `noniid_dirichlet`."

        clientTrainSets = sample.noniid_dirichlet_equal_split(
            num_clients, trainSet, alpha=cfg.alpha, num_classes=cfg.num_classes
        )
        clientTestSets = sample.noniid_dirichlet_equal_split(
            num_clients, testSet, alpha=cfg.alpha, num_classes=cfg.num_classes
        )
    else:
        exit(f"[ERROR] Illegal sample method: {type}.")

    return (clientTrainSets, clientTestSets)

# -----------------------------------------------------------------------------
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
        return self.dataset[self.idx[item]]

# -----------------------------------------------------------------------------
