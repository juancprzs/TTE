#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[1-500]
#SBATCH -J ImageNet_SGV
#SBATCH -o logs/ImageNet_SGV.%J.out
#SBATCH -e logs/ImageNet_SGV.%J.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=alfarrm@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH -A conf-gpu-2020.11.23
#SBATCH --constraint=ref_32T

source activate upd_pt

nvidia-smi

python main.py \
--checkpoint runs/baseline \
--num-chunk ${SLURM_ARRAY_TASK_ID} \
--chunks 500 --eps 0.00784 --experiment imagenet_nominal_training
