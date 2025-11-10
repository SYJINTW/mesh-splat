#!/bin/bash
# set -e ; echo "set -e activated: exit on error"

# === [DEBUGGING SCRIPT] ===========================
# This script serves as fast debugging of the whole pipeline (train+render+metrics)
# with a specific scene, policy, and budget.



# === [CONFIGS] ===========================
export CUDA_VISIBLE_DEVICES=2

BUDGET=(1572865) # arbitrary budget 
# BUDGET=(131072) # arbitrary budget 
# UNIT_BUDGET=1.6 # budget proportional to number of triangles

# POLICIES=("planarity" "area" "distortion" "uniform" "random") 
POLICY=("distortion") 


DATASET_DIR="/mnt/data1/syjintw/NEU/dataset/bicycle" 

SCENE_NAME="bicycle" # or other NeRF scene name
MESH_TYPE="colmap" # "sugar" or "colmap"
# MESH_FILE="$DATASET_DIR/colmap_mesh.ply" # "mesh.obj" for sugar, "mesh.ply" for colmap
MESH_FILE="/mnt/data1/syjintw/NEU/dataset/colmap/bicycle/checkpoint/mesh.ply"



RESOLUTION="" # or "--resolution 4" for faster debugging
IS_WHITE_BG="-w" # set to "--white_background" if the dataset has white background

DATE_TODAY=$(date +"%m%d")
SAVE_DIR="output/${DATE_TODAY}/Debug_${SCENE_NAME}_${MESH_TYPE}_${POLICY}_${BUDGET}"
LOG_FILE="pipeline-${DATE_TODAY}.log"

# === [MAIN SCRIPT] ===========================

echo > "$LOG_FILE" # clear log file


{
    echo "================================================================="
    echo "Starting training of Meshsplat"
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
    echo "Step 0/3: Warm Up"
    python train.py --eval \
    --warmup_only \
    -s "$DATASET_DIR" \
    -m "$SAVE_DIR" \
    --texture_obj_path "$MESH_FILE" \
    --mesh_type "$MESH_TYPE" \
    --debugging \
    --occlusion \
    --total_splats "$BUDGET"\
    --alloc_policy "$POLICY" \
    --gs_type gs_mesh \
    $IS_WHITE_BG \
    --iteration 10 \
    "$RESOLUTION"



    # --budget_per_tri "$UNIT_BUDGET" \
} | tee -a "$LOG_FILE"



{
    echo "Step 1/3: Training"
    python train.py --eval \
    -s "$DATASET_DIR" \
    -m "$SAVE_DIR" \
    --texture_obj_path "$MESH_FILE" \
    --mesh_type "$MESH_TYPE" \
    --debugging \
    --occlusion \
    --total_splats "$BUDGET"\
    --alloc_policy "$POLICY" \
    --policy_path "${SAVE_DIR}/${policy}_${budget}.npy" \
    --gs_type gs_mesh \
    $IS_WHITE_BG \
    --iteration 10 \
    "$RESOLUTION"




    # --budget_per_tri "$UNIT_BUDGET" \
} | tee -a "$LOG_FILE"



{
echo "Step 2/3: Rendering with the trained model"
python render_mesh_splat.py \
    -m "$SAVE_DIR" \
    --gs_type gs_mesh \
    --skip_train \
    --occlusion \
    --total_splats "$BUDGET"\
    --alloc_policy "$POLICY" \
    --texture_obj_path "$MESH_FILE" \
    --mesh_type "$MESH_TYPE" \
    --policy_path "${SAVE_DIR}/${policy}_${budget}.npy" \
    "$RESOLUTION"

    # --budget_per_tri "$UNIT_BUDGET" \
} | tee -a "$LOG_FILE"



{
echo "Step 3/3: Metrics computation"

python metrics.py \
    -m "$SAVE_DIR" \
    --gs_type gs_mesh
} | tee -a "$LOG_FILE"



cp $LOG_FILE "$SAVE_DIR/$LOG_FILE"
echo "Log file also saved to $SAVE_DIR/$LOG_FILE"
echo "Testing/Debugging completed!"