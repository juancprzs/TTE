#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[2-317]
#SBATCH -J chnkd_fc1_in
#SBATCH -o logs/chnkd_fc1_in.%J.out
#SBATCH -e logs/chnkd_fc1_in.%J.err
#SBATCH --time=4:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --constraint=ref_32T
#SBATCH -A conf-gpu-2020.11.23

conda activate guill
module load cuda/10.0.130

python autoattack-wrapper.py \
--flip \
--n-crops 1 \
--flip-crop \
--total-chunks 317 \
--load ~/imagenet_weights/feature_denoising_R152-Denoise.npz \
--data /local/reference/CV/ILSVR/classification-localization/data/jpeg \
--output-path /ibex/scratch/perezjc/TAR_imagenet/flip_crops1 \
--actual-chunk ${SLURM_ARRAY_TASK_ID} \
--save-adv
