<div align="center">
  <h1 align="center">FSL</h1>
</div>

## Introduction

Placeholder: Our FSL paper builds upon the code developed from the CSE-FL paper, noted below.

## How to run
* Cifar
```
python cse_fsl_main.py --dataset cifar -seed 200 -K 5 -U 5 -B 50 -E 1 --lr 0.15 --iid --gpu --test_round 1 --round 500 --save
```
* Femnist
```
python cse_fsl_main.py --dataset femnist -seed 200 -K 3500 -U 5 -B 10 -E 1 --lr 0.03 --noniid --gpu --test_round 1 --round 500 --save
```

## Citation

If you use this code in your research, please cite this paper.

```
@article{mu2023communication,
  title={Communication and Storage Efficient Federated Split Learning},
  author={Mu, Yujia and Shen, Cong},
  journal={arXiv preprint arXiv:2302.05599},
  year={2023}
}
```
# FSL