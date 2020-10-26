GPU=1
# CUDA_VISIBLE_DEVICES=$GPU python eval_wrapper.py --load weights/R152-Denoise.npz --attack-iter 0 --arch ResNetDenoise --seed 0 --exp-name baseline.txt
# CUDA_VISIBLE_DEVICES=$GPU python eval_wrapper.py --load weights/R152-Denoise.npz --attack-iter 0 --arch ResNetDenoise --seed 0 --exp-name flip.txt --flip
# CUDA_VISIBLE_DEVICES=$GPU python eval_wrapper.py --load weights/R152-Denoise.npz --attack-iter 0 --arch ResNetDenoise --seed 0 --exp-name crop1-0.txt --n-crops 1
# CUDA_VISIBLE_DEVICES=$GPU python eval_wrapper.py --load weights/R152-Denoise.npz --attack-iter 0 --arch ResNetDenoise --seed 1 --exp-name crop1-1.txt --n-crops 1
# CUDA_VISIBLE_DEVICES=$GPU python eval_wrapper.py --load weights/R152-Denoise.npz --attack-iter 0 --arch ResNetDenoise --seed 2 --exp-name crop1-2.txt --n-crops 1

# CUDA_VISIBLE_DEVICES=$GPU python eval_wrapper.py --load weights/R152-Denoise.npz --attack-iter 0 --arch ResNetDenoise --seed 0 --exp-name crop2-0.txt --n-crops 2
# CUDA_VISIBLE_DEVICES=$GPU python eval_wrapper.py --load weights/R152-Denoise.npz --attack-iter 0 --arch ResNetDenoise --seed 1 --exp-name crop2-1.txt --n-crops 2


# CUDA_VISIBLE_DEVICES=$GPU python eval_wrapper.py --load weights/R152-Denoise.npz --attack-iter 0 --arch ResNetDenoise --seed 0 --exp-name flip-and-crop2-0.txt --n-crops 2 --flip
# CUDA_VISIBLE_DEVICES=$GPU python eval_wrapper.py --load weights/R152-Denoise.npz --attack-iter 0 --arch ResNetDenoise --seed 1 --exp-name flip-and-crop2-1.txt --n-crops 2 --flip
# CUDA_VISIBLE_DEVICES=$GPU python eval_wrapper.py --load weights/R152-Denoise.npz --attack-iter 0 --arch ResNetDenoise --seed 2 --exp-name flip-and-crop2-2.txt --n-crops 2 --flip

# CUDA_VISIBLE_DEVICES=$GPU python eval_wrapper.py --load weights/R152-Denoise.npz --attack-iter 0 --arch ResNetDenoise --seed 0 --exp-name flip-and-crop3-0.txt --n-crops 3 --flip
# CUDA_VISIBLE_DEVICES=$GPU python eval_wrapper.py --load weights/R152-Denoise.npz --attack-iter 0 --arch ResNetDenoise --seed 1 --exp-name flip-and-crop3-1.txt --n-crops 3 --flip
# CUDA_VISIBLE_DEVICES=$GPU python eval_wrapper.py --load weights/R152-Denoise.npz --attack-iter 0 --arch ResNetDenoise --seed 2 --exp-name flip-and-crop3-2.txt --n-crops 3 --flip

CUDA_VISIBLE_DEVICES=$GPU python eval_wrapper.py --load weights/R152-Denoise.npz --attack-iter 0 --arch ResNetDenoise --seed 0 --exp-name flip-and-crop1-0.txt --n-crops 1 --flip --flip-crop
CUDA_VISIBLE_DEVICES=$GPU python eval_wrapper.py --load weights/R152-Denoise.npz --attack-iter 0 --arch ResNetDenoise --seed 1 --exp-name flip-and-crop1-1.txt --n-crops 1 --flip --flip-crop
CUDA_VISIBLE_DEVICES=$GPU python eval_wrapper.py --load weights/R152-Denoise.npz --attack-iter 0 --arch ResNetDenoise --seed 2 --exp-name flip-and-crop1-2.txt --n-crops 1 --flip --flip-crop

CUDA_VISIBLE_DEVICES=$GPU python eval_wrapper.py --load weights/R152-Denoise.npz --attack-iter 0 --arch ResNetDenoise --seed 0 --exp-name flip-and-crop4-0.txt --n-crops 4 --flip --flip-crop
CUDA_VISIBLE_DEVICES=$GPU python eval_wrapper.py --load weights/R152-Denoise.npz --attack-iter 0 --arch ResNetDenoise --seed 1 --exp-name flip-and-crop4-1.txt --n-crops 4 --flip --flip-crop
CUDA_VISIBLE_DEVICES=$GPU python eval_wrapper.py --load weights/R152-Denoise.npz --attack-iter 0 --arch ResNetDenoise --seed 2 --exp-name flip-and-crop4-2.txt --n-crops 4 --flip --flip-crop