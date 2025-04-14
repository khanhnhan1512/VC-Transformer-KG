#!/bin/bash
#SBATCH --job-name=vc_vmp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --output=/media02/lnthanh01/vmphat/logs/vc_vmp/%j-%x.out
#SBATCH --error=/media02/lnthanh01/vmphat/logs/vc_vmp/%j-%x.err
#SBATCH --mail-user=vmphat21@clc.fitus.edu.vn

# Load cuda
# spack load cuda@11.7.0
spack load cuda@11.8.0

# Config conda
eval "$(conda shell.bash hook)"
conda activate vc_vmp

# Change dir
cd /media02/lnthanh01/vmphat/VC-Transformer-KG/

python train.py
