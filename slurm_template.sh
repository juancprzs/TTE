#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[0-80]
#SBATCH -J abl_chunk1
#SBATCH -o logs/abl_chunk1.%J.out
#SBATCH -e logs/abl_chunk1.%J.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH -A conf-gpu-2020.11.23

ind=${SLURM_ARRAY_TASK_ID};
val1=$(($ind / 9)); val1=$(($val1 + 1));
val2=$(($ind % 9)); val2=$(($val2 + 1));

python main.py \
--chunks 2 \
--checkpoint runs/only_crop_x$val1-y$val2 \
--experiment local_trades \
--n-crops 1 \
--num-chunk 1 \
--x_coord $val1 \
--y_coord $val2
