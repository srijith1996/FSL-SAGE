# -----------------------------------------------------------------------------
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from datasets import sample, femnist

# -----------------------------------------------------------------------------
def get_dataset(cfg):
    '''
        Download the dataset (if needed) and depart it for clients.

        Returns:
            trainSet:	The whole training set
            testSet:	The whole test set
            userGroup:	The sample indexs of each client (dict.)
    '''

    if cfg.name == 'cifar10':
        dataDir = '../datas/cifar'
        
        #random.seed(10)
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

    elif cfg.name == 'imagenet':
        dataDir = '../datas/imagenet'

        transform = transforms.Compose([
             transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #,
        ])

        trainSet = datasets.ImageFolder(root=dataDir + '/train', transform=transform)
        testSet = datasets.ImageFolder(root=dataDir + '/val', transform=transform)

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
        inputs, labels = self.dataset[self.idx[item]]
        return inputs, labels

# -----------------------------------------------------------------------------
