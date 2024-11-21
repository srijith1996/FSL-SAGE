# Train model on CIFAR dataset
python FSL_main_linear.py --dataset cifar -seed 200  \
    -K 3 -L 10 -U 3 -B 128 -E 1 -batch_round 5       \
    --server-lr 0.001 --client-lr 0.001            \
    --iid --gpu --round 200 --save

#python FSL_main_linear.py --dataset cifar -seed 200 -K 5 -L 2 -U 5 -B 50 -E 1 --lr 0.15 --noniid --gpu --test_round 1 --round 100 --save --shard 6
# python FSL_main_linear.py --dataset cifar -seed 128 -K 5 -L 10 -U 5 -B 50 -E 1 --lr 0.15 --noniid --gpu --test_round 1 --round 100 --save --shard 6
