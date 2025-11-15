#!/bin/bash
# set -e ; echo "set -e activated: exit on error"

# === [ORIGINAL GS SCRIPT] ===========================

# run shell_script/rename_gs_dir.sh to get proper folder structure


# === [CONFIGS] ===========================
export CUDA_VISIBLE_DEVICES=3



SCENE_NAME="hotdog" # add a loop for multiple scenes if needed
DATASET_DIR="/mnt/data1/samk/NEU/dataset/${SCENE_NAME}" 


# MESH_TYPE="milo" # "sugar" or "colmap" or "milo"
# MESH_FILE="/mnt/data1/samk/NEU/dataset/milo_meshes/${SCENE_NAME}/downsample_50/mesh.ply"
# MESH_IMG_DIR=$(dirname "$MESH_FILE")

# ITERATION=7000
ITERATION=100
RESOLUTION="" # or "--resolution 4" for faster debugging
IS_WHITE_BG="-w" # set to "--white_background" if the dataset has white background

DATE_TODAY=$(date +"%m%d")
SAVE_DIR="output/hotdog_gs"
# POLICY_CACHED="${SAVE_DIR}/${POLICY}_${UNIT_BUDGET}.npy"
LOG_FILE="pipeline-${DATE_TODAY}.log"

# === [MAIN SCRIPT] ===========================

echo > "$LOG_FILE" # clear log file


{
    echo "================================================================="
    echo "Starting training of Meshsplat (Pure GS Version)"
    echo "Running on $(hostname), on branch $(git branch --show-current)"
    echo "Read dataset from: $DATASET_DIR"
    echo "Using mesh file: $MESH_FILE"
    echo "Mesh type: $MESH_TYPE"
    echo "White background: $IS_WHITE_BG (empty=NO)"
    echo "Output will be saved to: $SAVE_DIR"
    date +"%Y-%m-%d %H:%M:%S"
    echo "GPU cores: $CUDA_VISIBLE_DEVICES"
    echo "================================================================="
} | tee -a "$LOG_FILE"



{
echo "Step 2/3: Rendering with the trained model"
python render_gs.py \
    -m "$SAVE_DIR" \
    --gs_type gs \
    --skip_train \
    $RESOLUTION \
    $IS_WHITE_BG \
    2>&1

    # --budget_per_tri "$UNIT_BUDGET" \
} | tee -a "$LOG_FILE"



{
echo "Step 3/3: Metrics computation"

python metrics.py \
    -m "$SAVE_DIR" \
    --gs_type gs \
    2>&1
} | tee -a "$LOG_FILE"



cp $LOG_FILE "$SAVE_DIR/$LOG_FILE"
echo "Log file also saved to $SAVE_DIR/$LOG_FILE"
echo "Testing/Debugging completed!"