# Train model on CIFAR dataset
#python ecse_fsl_main.py --dataset cifar -seed 200 -K 5 -U 5 -B 50 -E 1 --lr 0.15 --iid --gpu --test_round 1 --round 500 --save
#python ecse_fsl_main.py --dataset cifar -seed 200 -K 5 -L 2 -U 5 -B 50 -E 1 --lr 0.15 --noniid --gpu --test_round 1 --round 100 --save --shard 6
python FSL_main_linear.py --dataset cifar -seed 128 -K 5 -L 10 -U 5 -B 50 -E 1 --lr 0.15 --noniid --gpu --test_round 1 --round 100 --save #--shard 6
