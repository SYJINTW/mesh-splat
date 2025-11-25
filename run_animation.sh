#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# ======= Config ======
SCENE_NAME="hotdog"
DATASET_DIR="/mnt/data1/samk/NEU/dataset/${SCENE_NAME}"
MESH_TYPE="colmap"
MESH_FILE="/mnt/data1/samk/NEU/dataset/${SCENE_NAME}/colmap_mesh/mesh.ply"
MESH_IMG_DIR=$(dirname "$MESH_FILE")

MODEL_PATH="output/1114_hotdog_colmap/hotdog/area_148783_occlusion"
POLICY_CACHED="${MODEL_PATH}/area_148783.npy"

ITERATION="7000"
BUDGET="148783"
POLICY="area"

# Animation settings
TRANSFORM="hotdog_fly"  # choices: ficus_sinus, hotdog_fly, ficus_pot, ship_sinus, make_smaller, none

IS_WHITE_BG=""  # set to "--white_background" if the dataset has white background
RESOLUTION=""   # or "--resolution 4" for faster debugging

# ======= Run Animation ======
python render_mesh_splat_animated.py \
    -m "$MODEL_PATH" \
    -s "$DATASET_DIR" \
    --gs_type gs_mesh \
    --skip_train \
    --occlusion \
    --total_splats "$BUDGET" \
    --alloc_policy "$POLICY" \
    --texture_obj_path "$MESH_FILE" \
    --mesh_type "$MESH_TYPE" \
    --policy_path "$POLICY_CACHED" \
    --precaptured_mesh_img_path "$MESH_IMG_DIR" \
    --transform "$TRANSFORM" \
    --iteration "$ITERATION" \
    $IS_WHITE_BG \
    $RESOLUTION

echo "Animation rendering completed!"
echo "Results saved to: ${MODEL_PATH}/test/ours_${ITERATION}/renders_animated_gs_mesh/"