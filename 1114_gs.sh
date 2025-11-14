#!/bin/bash
# set -e ; echo "set -e activated: exit on error"

# === [ORIGINAL GS SCRIPT] ===========================


# === [CONFIGS] ===========================
export CUDA_VISIBLE_DEVICES=2



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
SAVE_DIR="output/_DEBUG_${DATE_TODAY}/${SCENE_NAME}_GS_pure"
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


# {
#     echo "Step 0/3: Warm Up"
#     python train.py --eval \
#     --warmup_only \
#     -s "$DATASET_DIR" \
#     -m "$SAVE_DIR" \
#     --texture_obj_path "$MESH_FILE" \
#     --mesh_type "$MESH_TYPE" \
#     --debugging \
#     --occlusion \
#     --budget_per_tri "$UNIT_BUDGET" \
#     --alloc_policy "$POLICY" \
#     --gs_type gs_mesh \
#     --policy_path "$POLICY_CACHED" \
#     --precaptured_mesh_img_path "$MESH_IMG_DIR" \
#     $IS_WHITE_BG \
#     $RESOLUTION \
#     --iteration 10 \
#     2>&1
# } | tee -a "$LOG_FILE"



{
    echo "Step 1/3: Training"
    python train.py --eval \
    -s "$DATASET_DIR" \
    -m "$SAVE_DIR" \
    --debugging \
    --debug_freq 10 \
    --gs_type gs \
    $IS_WHITE_BG \
    $RESOLUTION \
    --iteration $ITERATION \
    2>&1
    # --budget_per_tri "$UNIT_BUDGET" \
} | tee -a "$LOG_FILE"



{
echo "Step 2/3: Rendering with the trained model"
python render_mesh_splat.py \
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