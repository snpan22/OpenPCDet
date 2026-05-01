#!/bin/bash
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --job-name=cp_mf_waymo_pred_train_36
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=cp_mf_waymo_pred_val_36_%j.out
#SBATCH --error=cp_mf_waymo_pred_val_36_%j.err

module load anaconda3
source activate /storage/project/r-gchou3-0/spanse30/ptt_env

cd /storage/project/r-gchou3-0/spanse30/OpenPCDet
export PYTHONPATH=$PWD:$PYTHONPATH

torchrun --nproc_per_node=2 \
    tools/test.py \
    --launcher pytorch \
    --cfg_file tools/cfgs/custom_models/centerpoint_multiframe_waymo.yaml \
    --batch_size 8 \
    --workers 4 \
    --ckpt output/cfgs/custom_models/centerpoint_multiframe_waymo/default/ckpt/checkpoint_epoch_36.pth \
    --set DATA_CONFIG.DATA_SPLIT.test val