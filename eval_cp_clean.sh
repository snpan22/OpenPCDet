#!/bin/bash
#SBATCH --account=gts-gchou3-ideasci23_dgx 
#SBATCH --job-name=eval_centerpoint_clean
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Generate timestamp (YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create output folder
RESULTS_DIR=/storage/scratch1/9/spanse30/PTT/metrics
# mkdir -p $RESULTS_DIR


module load anaconda3
eval "$(conda shell.bash hook)"
conda activate /storage/project/r-gchou3-0/spanse30/ptt_env

cd /storage/project/r-gchou3-0/spanse30/OpenPCDet
export PYTHONPATH=$PWD:$PYTHONPATH

srun python eval_cp_clean.py \
    --cfg_file 'tools/cfgs/waymo_models/centerpoint.yaml'  \
    --ckpt output/cfgs/custom_models/centerpoint_singleframe_waymo/default/ckpt/checkpoint_epoch_30.pth \
    --log_file $RESULTS_DIR/cp_clean_${TIMESTAMP}.log \
    --metrics_out $RESULTS_DIR/cp_clean_${TIMESTAMP}.json \
    > $RESULTS_DIR/slurm_cp_clean_${TIMESTAMP}.out \
    2> $RESULTS_DIR/slurm_cp_clean_${TIMESTAMP}.err
