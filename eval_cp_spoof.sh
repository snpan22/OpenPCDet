#!/bin/bash
#SBATCH --account=gts-gchou3-ideas_l40s
#SBATCH --job-name=ptt_pla_easy
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=07:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Generate timestamp (YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create output folder
RESULTS_DIR=/storage/scratch1/9/spanse30/PTT/pla/metrics_pi
# mkdir -p $RESULTS_DIR




module load anaconda3
eval "$(conda shell.bash hook)"
conda activate /storage/project/r-gchou3-0/spanse30/ptt_env

cd /storage/project/r-gchou3-0/spanse30/OpenPCDet
export PYTHONPATH=$PWD:$PYTHONPATH

srun python eval_cp_spoof_pi.py \
    --cfg_file /storage/project/r-gchou3-0/spanse30/OpenPCDet/tools/cfgs/waymo_models/centerpoint.yaml  \
    --ckpt /storage/project/r-gchou3-0/spanse30/OpenPCDet/output/cfgs/custom_models/centerpoint_singleframe_waymo/default/ckpt/checkpoint_epoch_30.pth \
    --dataset /storage/scratch1/9/spanse30/PTT/pla/datasets/global_easy \
    --pred_dir /storage/scratch1/9/spanse30/PTT/pla/preds_cp/global_easy \
    --log_file $RESULTS_DIR/cp_global_easy_eval${TIMESTAMP}.log \
    --metrics_out $RESULTS_DIR/cp_global_easy.json \
    --asr_path /storage/scratch1/9/spanse30/PTT/pla/asr/cp_global_easy.pkl \
    > $RESULTS_DIR/slurm_cp_global_easy_eval_${TIMESTAMP}.out \
    2> $RESULTS_DIR/slurm_cp_global_easy_eval_${TIMESTAMP}.err

#SBATCH --account=gts-gchou3-ideasci23_dgx 
#SBATCH --job-name=eval_cp
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1