#!/bin/bash
# set -e ; echo "set -e activated: exit on error"

# === [DEBUGGING SCRIPT] ===========================
# This script serves as fast debugging of the whole pipeline (train+render+metrics)
# with a specific scene, policy, and budget.



# === [CONFIGS] ===========================
export CUDA_VISIBLE_DEVICES=0

# [TODO] make the pipeline support BUDGET=0 (mesh only) case
# BUDGET=(2000000) # arbitrary budget 
UNIT_BUDGET=0.5 # budget proportional to number of triangles


# POLICIES=("planarity" "area" "distortion" "uniform" "random", "mixed") 

# [DOING] test all combinations of mixed policy

# using min-max normalization 

# POLICY=("mixed22") done
# POLICY=("mixed13") done
# POLICY=("mixed31")
POLICY=("distortion")
# POLICY=("area") done


# using other normalization
# POLICY=("mixed_v2g2") 
# POLICY=("mixed_v1g3")
# POLICY=("mixed_v3g1")


SCENE_NAME="hotdog" # add a loop for multiple scenes if needed
DATASET_DIR="/mnt/data1/samk/NEU/dataset/${SCENE_NAME}" 
MESH_TYPE="milo" # "sugar" or "colmap" or "milo"
MESH_FILE="/mnt/data1/samk/NEU/dataset/milo_meshes/${SCENE_NAME}/${SCENE_NAME}.ply"




# DATASET_DIR="/mnt/data1/samk/NEU/dataset/bicycle" 

# SCENE_NAME="bicycle" # or other NeRF scene name
# MESH_TYPE="colmap" # "sugar" or "colmap"
# # SCENE_NAME="bicycle_ds_10"
# MESH_FILE="/mnt/data1/samk/NEU/dataset/colmap/bicycle/downsampled_10/mesh.ply"
# # MESH_FILE="/mnt/data1/samk/NEU/dataset/colmap/bicycle/downsampled_30/mesh.ply"

MESH_IMG_DIR=$(dirname "$MESH_FILE")


RESOLUTION="" # or "--resolution 4" for faster debugging
IS_WHITE_BG="-w" # set to "--white_background" if the dataset has white background
ITERATIONS=3000


DATE_TODAY=$(date +"%m%d")
SAVE_DIR="output/_DEBUG_${DATE_TODAY}/${SCENE_NAME}_${MESH_TYPE}_${POLICY}_${UNIT_BUDGET}"
POLICY_CACHED="${SAVE_DIR}/${POLICY}_${UNIT_BUDGET}.npy"
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
    --budget_per_tri "$UNIT_BUDGET" \
    --alloc_policy "$POLICY" \
    --gs_type gs_mesh \
    --precaptured_mesh_img_path "$MESH_IMG_DIR" \
    $IS_WHITE_BG \
    $RESOLUTION \
    --iteration 10 \
    2>&1
    
    
    # --policy_path "$POLICY_CACHED" \
} | tee -a "$LOG_FILE"



echo "Step 1/3: Training"
python train.py --eval \
    -s "$DATASET_DIR" \
    -m "$SAVE_DIR" \
    --texture_obj_path "$MESH_FILE" \
    --mesh_type "$MESH_TYPE" \
    --debugging \
    --debug_freq 500 \
    --occlusion \
    --budget_per_tri "$UNIT_BUDGET" \
    --alloc_policy "$POLICY" \
    --policy_path "$POLICY_CACHED" \
    --precaptured_mesh_img_path "$MESH_IMG_DIR" \
    --gs_type gs_mesh \
    $IS_WHITE_BG \
    $RESOLUTION \
    --iteration $ITERATIONS >> $LOG_FILE
    



{
echo "Step 2/3: Rendering with the trained model"
python render_mesh_splat.py \
    -m "$SAVE_DIR" \
    --gs_type gs_mesh \
    --skip_train \
    --occlusion \
    --budget_per_tri "$UNIT_BUDGET" \
    --alloc_policy "$POLICY" \
    --texture_obj_path "$MESH_FILE" \
    --mesh_type "$MESH_TYPE" \
    $RESOLUTION \
    $IS_WHITE_BG \
    --policy_path "$POLICY_CACHED" \
    2>&1

    # --budget_per_tri "$UNIT_BUDGET" \
} | tee -a "$LOG_FILE"



{
echo "Step 3/3: Metrics computation"

python metrics.py \
    -m "$SAVE_DIR" \
    --gs_type gs_mesh \
    2>&1
} | tee -a "$LOG_FILE"



cp $LOG_FILE "$SAVE_DIR/$LOG_FILE"
echo "Log file also saved to $SAVE_DIR/$LOG_FILE"
echo "Testing/Debugging completed!"