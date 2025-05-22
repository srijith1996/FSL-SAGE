DATASET=$1
DEVICE=$2
ALGOS=("fsl_sage" "cse_fsl")

for ALGO in ${ALGOS[@]}; do
    python main.py -m model=resnet18 algorithm=$ALGO dataset=$DATASET dataset.distribution=iid project=fsl_sage_abalate_model_sizes model.auxiliary.name=resnet18-linear,resnet18-half,resnet18,resnet18-full rounds=9999999 device=$DEVICE
done

