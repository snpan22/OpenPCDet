#!/bin/bash
#SBATCH --account=gts-gchou3
#SBATCH --job-name=waymo_36_pred
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:2
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=waymo_36_pred_val_a100%j.out
#SBATCH --error=waymo_36_pred_val_a100_%j.err

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