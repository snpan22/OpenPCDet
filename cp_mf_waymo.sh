#!/bin/bash
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --job-name=cp_mf_waymo
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:8
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --output=cp_sf_waymo_%j.out
#SBATCH --error=cp_sf_waymo_%j.err

module load anaconda3
source activate /storage/project/r-gchou3-0/spanse30/ptt_env

cd /storage/project/r-gchou3-0/spanse30/OpenPCDet
export PYTHONPATH=$PWD:$PYTHONPATH

torchrun --nproc_per_node=8 \
    tools/train.py \
    --launcher pytorch \
    --cfg_file tools/cfgs/custom_models/centerpoint_multiframe_waymo.yaml \
    --batch_size 8 \
    --workers 4 \
    --ckpt output/cfgs/custom_models/centerpoint_multiframe_waymo/default/ckpt/checkpoint_epoch_29.pth