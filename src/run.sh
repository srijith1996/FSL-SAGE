# Train model on CIFAR dataset
python FSL_main_linear.py --dataset cifar -seed 200  \
    -K 2 -L 1 -U 2 -B 100 -E 1 -batch_round 5       \
    --server-lr 0.005 --client-lr 0.0005             \
    --iid --gpu --round 50 --save

#python FSL_main_linear.py --dataset cifar -seed 200 -K 5 -L 2 -U 5 -B 50 -E 1 --lr 0.15 --noniid --gpu --test_round 1 --round 100 --save --shard 6
# python FSL_main_linear.py --dataset cifar -seed 128 -K 5 -L 10 -U 5 -B 50 -E 1 --lr 0.15 --noniid --gpu --test_round 1 --round 100 --save --shard 6
