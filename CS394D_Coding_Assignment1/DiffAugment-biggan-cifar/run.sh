
sbatch <<- EOF
	#!/bin/bash
	#SBATCH -N 1
	#SBATCH -J MyJob
	#SBATCH -o slurm/%J.out
	#SBATCH -e slurm/%J.err
	#SBATCH --time=1-00:00:00
	#SBATCH --mem=100G
	#SBATCH --gres=gpu:2
	#SBATCH --cpus-per-task=16
	#SBATCH --partition batch
	#run the application:
	module load gcc/6.4.0

	CUDA_VISIBLE_DECIVES="0,1" python -u train.py \
	--experiment_name DiffAugment-biggan-cifar10-0.05-diffaugment_all-CR_10 \
	--DiffAugment translation,cutout,color\
	--CR_augment translation,cutout,color \
	--CR 10\
	--mirror_augment --use_multiepoch_sampler \
	--which_best FID --num_inception_images 10000 \
	--shuffle --batch_size 50 --parallel \
	--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 5000 --num_samples 2500 \
	--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
	--dataset C10 \
	--G_ortho 0.0 \
	--G_attn 0 --D_attn 0 \
	--G_init N02 --D_init N02 \
	--ema --use_ema --ema_start 1000 \
	--test_every 4000 --save_every 2000 --seed 0

	module unload gcc/6.4.0
EOF
