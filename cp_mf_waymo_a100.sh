#!/bin/bash
#SBATCH --account=gts-gchou3
#SBATCH --job-name=cp_mf_waymo_a100_2gpu
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=cp_mf_waymo_a100_%j.out
#SBATCH --error=cp_mf_waymo_a100_%j.err

module load anaconda3
source activate /storage/project/r-gchou3-0/spanse30/ptt_env

cd /storage/project/r-gchou3-0/spanse30/OpenPCDet
export PYTHONPATH=$PWD:$PYTHONPATH

torchrun --nproc_per_node=2 \
    tools/train.py \
    --launcher pytorch \
    --cfg_file tools/cfgs/custom_models/centerpoint_multiframe_waymo.yaml \
    --batch_size 4 \
    --workers 4 \
    --ckpt output/cfgs/custom_models/centerpoint_multiframe_waymo/default/ckpt/checkpoint_epoch_15.pth


# #!/bin/bash
# #SBATCH --account=gts-gchou3
# #SBATCH --job-name=cp_mf_waymo_a100_8gpu
# #SBATCH --partition=gpu-a100
# #SBATCH --gres=gpu:a100:8
# #SBATCH --nodes=1
# #SBATCH --ntasks=8
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=160G
# #SBATCH --time=24:00:00
# #SBATCH --output=cp_mf_waymo_a100_%j.out
# #SBATCH --error=cp_mf_waymo_a100_%j.err

# module load anaconda3
# source activate /storage/project/r-gchou3-0/spanse30/ptt_env

# cd /storage/project/r-gchou3-0/spanse30/OpenPCDet
# export PYTHONPATH=$PWD:$PYTHONPATH

# torchrun --nproc_per_node=8 \
#     tools/train.py \
#     --launcher pytorch \
#     --cfg_file tools/cfgs/custom_models/centerpoint_multiframe_waymo.yaml \
#     --batch_size 8 \
#     --workers 4 \
#     --ckpt output/cfgs/custom_models/centerpoint_multiframe_waymo/default/ckpt/checkpoint_epoch_15.pth
