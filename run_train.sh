

# === [DEBUGGING SCRIPT] ===========================
# This script serves as fast debugging for training with a specific scene, policy, and budget.
set -e
CUDA_VISIBLE_DEVICES=0



# BUDGETS=(16384 23170 32768 46341 65536 92682 131072 185364 262144 368589 524288) # Add your budgets here
# BUDGET=(1572865) # Add your budgets here
UNIT_BUDGET=2.2

# POLICIES=("planarity" "area" "uniform" "random") # Add your policies here
POLICY=("distortion") # Add your policies here

LOG_FILE="train.log"
SCENE_NAME="hotdog" # or other NeRF scene name
DATASET_DIR="/mnt/data1/syjintw/NEU/dataset/hotdog"
SAVE_DIR="output/Debug_${SCENE_NAME}_${POLICY}_${UNIT_BUDGET}"



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
--texture_obj_path "$DATASET_DIR/mesh.obj" \
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
    --texture_obj_path "$DATASET_DIR/mesh.obj" \
    --policy_path "${SAVE_DIR}/${policy}_${budget}.npy" >> "$LOG_FILE"




# --total_splats "$BUDGET" \
# --policy_path "$DATASET_DIR/policy/${POLICY}_${BUDGET}.npy" \


cp $LOG_FILE "$SAVE_DIR/$LOG_FILE"
echo "Log file also saved to $SAVE_DIR/$LOG_FILE"
echo "Testing/Debugging completed!"