#!/bin/bash
#SBATCH --job-name=hyper_search
#SBATCH --output=/home/s2186747/git/ug-project/logs/hyper_%x_%j.out
#SBATCH --error=/home/s2186747/git/ug-project/logs/hyper_%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=16000
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclusive

set -x
echo "Job running on ${SLURM_JOB_NODELIST}"
TIMESTAMP=$(date +'%Y-%m-%d_%H-%M-%S')
echo "Job started: $TIMESTAMP"

source ~/.bashrc
set -e
echo "Activating conda environment: toybox"
source /home/s2186747/miniconda3/etc/profile.d/conda.sh
conda activate toybox

BASE_OUTPUT_DIR="/home/s2186747/git/ug-project/output"
OUTPUT_NAME="hyperparameter_search_${SLURM_JOB_ID}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${OUTPUT_NAME}"
mkdir -p "$OUTPUT_DIR"
echo "Outputs will be saved to: $OUTPUT_DIR"

SCRATCH_DISK=/disk/scratch
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
SCRATCH_OUTPUT_DIR="${SCRATCH_HOME}/output/hyperparameter_search_${TIMESTAMP}"
mkdir -p "$SCRATCH_OUTPUT_DIR"
echo "Scratch output directory created at: $SCRATCH_OUTPUT_DIR"

SCRATCH_DATA_DIR="/${SCRATCH_HOME}/project_data"
mkdir -p $SCRATCH_DATA_DIR
echo "Copying data to scratch disk..."
rsync --archive --update --compress --progress --exclude='/afs' --exclude='/proc' "/home/s2186747/data/project/Toybox/" "$SCRATCH_DATA_DIR/"
echo "Data copied to scratch disk."

python /home/s2186747/git/ug-project/scripts/hyperparameter_search.py --data_dir "$SCRATCH_DATA_DIR" --epochs 10 --output_dir "$SCRATCH_OUTPUT_DIR"


echo "Copying outputs from scratch to final output directory..."
rsync --archive --update --compress --progress "$SCRATCH_HOME/output/" "$BASE_OUTPUT_DIR/"
echo "Outputs copied back to distributed filesystem."

echo "Job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
