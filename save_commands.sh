# Centralized version of resnet on imagenet
python test/resnet_imagenet_local.py -a resnet50 datas/imagenet --epochs 135

# Test 'centralized' (mock with fedavg) version of resnet152
python main.py model=resnet56_cifar100 algorithm=fed_avg dataset=cifar100 num_clients=1 device="cuda:1"

# Train multiple algs
python main.py -m device="cuda:2" model=resnet152 algorithm=fed_avg,sl_single_server,sl_multi_server,cse_fsl,fsl_sage dataset=cifar100 