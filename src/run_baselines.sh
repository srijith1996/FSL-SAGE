python baselines.py --dataset cifar -seed 200   \
    -K 3 -L 2 -U 3 -B 128 -E 1 -batch_round 130 \
    --server-lr 0.003 --client-lr 0.0002        \
    --iid --gpu --round 200 --save

