#!/bin/bash
#SBATCH --job-name=toybox_model
#SBATCH --output=/home/s2186747/git/ug-project/logs/%x_%j.out  # Log file named by job and job ID
#SBATCH --error=/home/s2186747/git/ug-project/logs/%x_%j.err   # Error file
#SBATCH --time=08:00:00                                     # Time limit
#SBATCH --mem=14000                                         # Memory allocation
#SBATCH --cpus-per-task=2                                   # Number of CPUs
#SBATCH --gres=gpu:1                                        # Request GPU
#SBATCH --nodes=1                                           # Number of nodes
#SBATCH --exclusive

if [[ "$1" == "--help" ]]; then
    echo "Usage: sbatch your_script.sbatch [options]"
    echo ""
    echo "Options:"
    echo "  --model               Model name (e.g., MLP1layer, MLP2layer, MLP3layer, AlexNet)"
    echo "  --dataset             Dataset name (e.g., toybox, toybox_grayscale, toybox_random_color, MNIST)"
    echo "  --epochs              Total number of training epochs"
    echo "  --learning_rate       Learning rate for optimizer"
    echo "  --source_data_dir     Path to the dataset directory"
    echo "  --continue_from_epoch Epoch to continue training from (optional)"
    echo "  --source_output_dir   Path to directory where previously trained model is stored (optional)"
    echo "  --backbone_path       Path to weights of previously trained model (optional: only applies when continue_from_epoch=0)"
    echo ""
    echo "Example:"
    echo "  sbatch --time=02:00:00 --gres=gpu:1 scripts/run_job.sbatch \\"
    echo "    --model AlexNet \\"
    echo "    --dataset toybox \\"
    echo "    --epochs 10 \\"
    echo "    --learning_rate 0.01 \\"
    echo "    --source_data_dir /home/s2186747/data/project/toybox_sample_resize \\"
    echo "    --continue_from_epoch 5 \\"
    echo "    --source_output_dir /home/s2186747/git/ug-project/output/AlexNet_toybox_grayscale_2024-11-28_18-40-45"
    exit 0
fi

# debugging
set -x

echo "Job running on ${SLURM_JOB_NODELIST}"
TIMESTAMP=$(date +'%Y-%m-%d_%H-%M-%S')
echo "Job started: $TIMESTAMP"

echo "Setting up bash enviroment"
source ~/.bashrc
set -e  # Exit on first error

echo "Activating conda environment: toybox"
source /home/s2186747/miniconda3/etc/profile.d/conda.sh
conda activate toybox

CONTINUE_FROM_EPOCH=0

# Argument Parsing
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL_NAME="$2"; shift ;;
        --dataset) DATASET_NAME="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --learning_rate) LEARNING_RATE="$2"; shift ;;
        --source_data_dir) SOURCE_DATA_DIR="$2"; shift ;;
        --continue_from_epoch) CONTINUE_FROM_EPOCH="$2"; shift ;;
        --source_output_dir) SOURCE_OUTPUT_DIR="$2"; shift ;;
        --backbone_path) BACKBONE_PATH="$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Debugging: Ensure all arguments were parsed correctly
echo "Parsed Arguments:"
echo "MODEL_NAME: $MODEL_NAME"
echo "DATASET_NAME: $DATASET_NAME"
echo "EPOCHS: $EPOCHS"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "SOURCE_DATA_DIR: $SOURCE_DATA_DIR"
echo "CONTINUE_FROM_EPOCH: $CONTINUE_FROM_EPOCH"
echo "SOURCE_OUTPUT_DIR: $SOURCE_OUTPUT_DIR"
echo "BACKBONE_PATH: $BACKBONE_PATH"

# Validate Arguments
if [ -z "$MODEL_NAME" ] || [ -z "$DATASET_NAME" ] || [ -z "$EPOCHS" ] || [ -z "$LEARNING_RATE" ] || [ -z "$SOURCE_DATA_DIR" ]; then
    echo "Error: One or more required arguments are missing."
    echo "Usage: sbatch your_script.sbatch --model <model_name> --dataset <dataset_name> --epochs <num_epochs> --learning_rate <lr> --source_data_dir <path> [--continue_from_epoch <previous_model_epoch> --source_output_dir <path> --BACKBONE_PATH <path>]"
    exit 1
fi

# Generate a timestamped output directory
BASE_OUTPUT_DIR="/home/s2186747/git/ug-project/output"
OUTPUT_NAME="${MODEL_NAME}_${DATASET_NAME}_${SLURM_JOB_ID}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${OUTPUT_NAME}"

# Create the output directory
mkdir -p "$OUTPUT_DIR"
echo "Outputs will be saved to: $OUTPUT_DIR"

# Define the scratch output directory
SCRATCH_DISK=/disk/scratch
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
SCRATCH_OUTPUT_DIR="${SCRATCH_HOME}/output/${DATASET_NAME}_${TIMESTAMP}"

# Create the scratch output directory
mkdir -p "$SCRATCH_OUTPUT_DIR"
echo "Scratch output directory created at: $SCRATCH_OUTPUT_DIR"

# If SOURCE_OUTPUT_DIR is provided and CONTINUE_FROM_EPOCH is specified, prepare for continuing from previous state
if [[ -n "$SOURCE_OUTPUT_DIR" && -n "$CONTINUE_FROM_EPOCH" && -d "$SOURCE_OUTPUT_DIR" ]]; then
    SCRATCH_PREVIOUS_OUTPUT_DIR="${SCRATCH_HOME}/previous_output/${DATASET_NAME}_${TIMESTAMP}"
    mkdir -p "$SCRATCH_PREVIOUS_OUTPUT_DIR"
    echo "Copying previous model outputs to scratch disk..."
    rsync --archive --update --compress --progress "$SOURCE_OUTPUT_DIR/" "$SCRATCH_PREVIOUS_OUTPUT_DIR/"
    echo "Previous model outputs copied to scratch disk."
    CONTINUE_DIR="$SCRATCH_PREVIOUS_OUTPUT_DIR"
elif [[ -n "$BACKBONE_PATH" && -f "$BACKBONE_PATH" ]]; then
    SCRATCH_PREVIOUS_OUTPUT_DIR="${SCRATCH_HOME}/previous_output/${DATASET_NAME}_${TIMESTAMP}"
    mkdir -p "$SCRATCH_PREVIOUS_OUTPUT_DIR"
    echo "Copying backbone checkpoint to scratch disk..."
    cp "$BACKBONE_PATH" "$SCRATCH_PREVIOUS_OUTPUT_DIR/"
    echo "Backbone checkpoint file copied."
    CONTINUE_DIR="$SCRATCH_PREVIOUS_OUTPUT_DIR"
    echo "Using pretrained backbone from: $BACKBONE_PATH"
else
    CONTINUE_DIR="$SCRATCH_OUTPUT_DIR"
    echo "Starting training without previous model state."
fi

# Determine what to pass as --backbone_path
if [[ -n "$BACKBONE_PATH" && -f "$BACKBONE_PATH" ]]; then
    PASS_BACKBONE="${CONTINUE_DIR}/${MODEL_NAME}_${DATASET_NAME}_final.pth"
else
    PASS_BACKBONE=""
fi

if [ -z "$SOURCE_DATA_DIR" ] || [ ! -d "$SOURCE_DATA_DIR" ]; then
    echo "Error: SOURCE_DATA_DIR is not set or does not exist."
    exit 1
fi

SCRATCH_DATA_DIR="/${SCRATCH_HOME}/project_data"

echo "Creating scratch data directory..."
mkdir -p $SCRATCH_DATA_DIR
echo "Copying data to scratch disk..."
rsync --archive --update --compress --progress --exclude='/afs' --exclude='/proc' "$SOURCE_DATA_DIR/" "$SCRATCH_DATA_DIR/"
echo "Data copied to scratch disk."

python /home/s2186747/git/ug-project/scripts/main.py \
    --model "$MODEL_NAME" \
    --dataset "$DATASET_NAME" \
    --epochs "$EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --output_dir "$CONTINUE_DIR" \
    --data_dir "$SCRATCH_DATA_DIR" \
    --continue_from_epoch "$CONTINUE_FROM_EPOCH" \
    --backbone_path "$PASS_BACKBONE"

echo "Copying outputs from scratch to final output directory..."
rsync --archive --update --compress --progress "$CONTINUE_DIR/" "$OUTPUT_DIR/"
echo "Outputs copied back to distributed filesystem."

# Archieve scratch directory

ARCHIVE_PATH="/home/${USER}/project/scratch_archives"
mkdir -p ${ARCHIVE_PATH}
ARCHIVE_FILE="${ARCHIVE_PATH}/project_data_${TIMESTAMP}.tar.gz"
echo "Archiving scratch directory to ${ARCHIVE_FILE}"
tar -czf ${ARCHIVE_FILE} -C ${SCRATCH_HOME} project_data

echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
