
set -e
CUDA_VISIBLE_DEVICES=0



# BUDGETS=(16384 23170 32768 46341 65536 92682 131072 185364 262144 368589 524288) # Add your budgets here
BUDGET=(185364) # Add your budgets here

# POLICIES=("planarity" "area" "uniform" "rand_uni") # Add your policies here
POLICY=("planarity") # Add your policies here

LOG_FILE="train.log"
SCENE_NAME="hotdog" # or other NeRF scene name
DATASET_DIR="/mnt/data1/syjintw/NEU/dataset/hotdog"
SAVE_DIR="output/${SCENE_NAME}_${POLICY}_${BUDGET}"



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


python train.py --eval \
-s  "$DATASET_DIR" \
-m "$SAVE_DIR" \
--texture_obj_path /mnt/data1/syjintw/NEU/dataset/hotdog/mesh.obj \
--debugging \
--occlusion \
--total_splats "$BUDGET" \
--alloc_policy "$POLICY" \
--gs_type gs_mesh -w --iteration 10 >> "$LOG_FILE"

# with/without occlusion


# then render

# then metrics


cp $LOG_FILE "$SAVE_DIR/$LOG_FILE"
echo "Log file also saved to $SAVE_DIR/$LOG_FILE"
echo "Experiment completed!"