# Train model on CIFAR dataset
python FSL_main_linear.py --dataset cifar -seed 200 \
    -K 10 -L 1 -U 10 -B 100 -E 1 -batch_round 20      \
    --server-lr 0.1 --client-lr 0.02               \
    --iid --gpu --test_round 2 --round 50 --save

#python FSL_main_linear.py --dataset cifar -seed 200 -K 5 -L 2 -U 5 -B 50 -E 1 --lr 0.15 --noniid --gpu --test_round 1 --round 100 --save --shard 6
# python FSL_main_linear.py --dataset cifar -seed 128 -K 5 -L 10 -U 5 -B 50 -E 1 --lr 0.15 --noniid --gpu --test_round 1 --round 100 --save --shard 6
