if [ "$1" == "fsl_sage" ]; then
    pyfile="FSL_SAGE_main.py"
elif [ "$1" == "cse_fsl" ]; then
    pyfile="CSE_FSL_main.py"
elif [ "$1" == "baselines" ]; then
    # for now, change the method in the first line of 'if __name__ ==
    # "__main__"' block within the baselines.py file
    pyfile="baselines.py"
else
    echo "Which method do I run? Choose from fsl_sage, cse_fsl, or baselines"
    exit
fi

# Train model on CIFAR dataset
python "$pyfile" --dataset cifar -seed 200  \
    -K 3 -L 10 -U 3 -B 128 -E 1 -batch_round $2      \
    --server-lr 0.001 --client-lr 0.001            \
    --iid --gpu --round $3 --save
#python FSL_main_linear.py --dataset cifar -seed 200 -K 5 -L 2 -U 5 -B 50 -E 1 --lr 0.15 --noniid --gpu --test_round 1 --round 100 --save --shard 6
# python FSL_main_linear.py --dataset cifar -seed 128 -K 5 -L 10 -U 5 -B 50 -E 1 --lr 0.15 --noniid --gpu --test_round 1 --round 100 --save --shard 6
