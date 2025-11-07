

# === [DEBUGGING SCRIPT] ===========================
# This script serves as fast debugging for training with a specific scene, policy, and budget.
set -e
CUDA_VISIBLE_DEVICES=0



# BUDGETS=(16384 23170 32768 46341 65536 92682 131072 185364 262144 368589 524288) # Add your budgets here
# BUDGET=(1572865) # Add your budgets here
UNIT_BUDGET=2.4

# POLICIES=("planarity" "area" "uniform" "random") # Add your policies here
# POLICY=("distortion") # Debugging
POLICY=("random") # Debugging

DATASET_DIR="/mnt/data1/samk/NEU/dataset/hotdog"

SCENE_NAME="hotdog" # or other NeRF scene name
MESH_TYPE="colmap" # "sugar" or "colmap"
MESH_FILE="$DATASET_DIR/colmap_mesh.ply" # "mesh.obj" for sugar, "mesh.ply" for colmap

SAVE_DIR="output/Debug_${SCENE_NAME}_${MESH_TYPE}_${POLICY}_${UNIT_BUDGET}"
LOG_FILE="train.log"



{
    echo "================================================================="
    echo "Starting training of Meshsplat"
    echo "Running on $(hostname), on branch $(git branch --show-current)"
    echo "Read dataset from: $DATASET_DIR"
    echo "Output will be saved to: $SAVE_DIR"
    date +"%Y-%m-%d %H:%M:%S"
    echo "GPU cores: $CUDA_VISIBLE_DEVICES"
    echo "================================================================="
} | tee "$LOG_FILE"


echo "Step 1/2: Training"

python train.py --eval \
-s  "$DATASET_DIR" \
-m "$SAVE_DIR" \
--texture_obj_path "$MESH_FILE" \
--mesh_type "$MESH_TYPE" \
--debugging \
--occlusion \
--budget_per_tri "$UNIT_BUDGET" \
--alloc_policy "$POLICY" \
--gs_type gs_mesh -w --iteration 10 >> "$LOG_FILE"


echo "Step 2/2: Rendering with the trained model"


python render_mesh_splat.py \
    -m "$SAVE_DIR" \
    --gs_type gs_mesh \
    --skip_train \
    --occlusion \
    --budget_per_tri "$UNIT_BUDGET" \
    --alloc_policy "$POLICY" \
    --texture_obj_path "$MESH_FILE" \
    --policy_path "${SAVE_DIR}/${policy}_${budget}.npy" >> "$LOG_FILE"




# --total_splats "$BUDGET" \
# --policy_path "$DATASET_DIR/policy/${POLICY}_${BUDGET}.npy" \


cp $LOG_FILE "$SAVE_DIR/$LOG_FILE"
echo "Log file also saved to $SAVE_DIR/$LOG_FILE"
echo "Testing/Debugging completed!"