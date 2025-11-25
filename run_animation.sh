#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# ======= Config ======
SCENE_NAME="hotdog"
DATASET_DIR="/mnt/data1/samk/NEU/dataset/${SCENE_NAME}"
MESH_TYPE="milo"
MESH_FILE="/mnt/data1/samk/NEU/dataset/milo_meshes/${SCENE_NAME}/${SCENE_NAME}.ply"
MESH_IMG_DIR=$(dirname "$MESH_FILE")

POLICY_CACHED="${MODEL_PATH}/area_140000.npy"

ITERATION="15000"
BUDGET="148783"
POLICY="area"

MODEL_PATH="output/1125_hotdog_milo/hotdog/${POLICY}_${BUDGET}_occlusion"

# Animation settings
TRANSFORM="hotdog_fly"  # choices: ficus_sinus, hotdog_fly, ficus_pot, ship_sinus, make_smaller, none

IS_WHITE_BG=""  # set to "--white_background" if the dataset has white background
RESOLUTION=""   # or "--resolution 4" for faster debugging

# ======= Run Animation ======
{
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
    --transform "$TRANSFORM" \
    --iteration "$ITERATION" \
    $IS_WHITE_BG \
    $RESOLUTION 2>&1 


    # --precaptured_mesh_img_path "$MESH_IMG_DIR" \
} | tee "animate.log" 



RENDER_DIR="${MODEL_PATH}/test/ours_${ITERATION}/renders_animated_gs_mesh"
VIDEO_PATH="${MODEL_PATH}/test/ours_${ITERATION}/animation.mp4"

echo "Animation rendering completed!"
echo "Results saved to: ${RENDER_DIR}"

if [ -d "$RENDER_DIR" ]; then
    echo "Combining frames into video..."
    # -y: overwrite output
    # -framerate 30: 30 fps
    # -i .../%05d.png: input pattern for 00000.png, 00001.png, etc.
    # -c:v libx264: H.264 codec
    # -pix_fmt yuv420p: pixel format for compatibility
    ffmpeg -y -framerate 30 -i "${RENDER_DIR}/%05d.png" -c:v libx264 -pix_fmt yuv420p "$VIDEO_PATH"
    echo "Video saved to: $VIDEO_PATH"
else
    echo "Render directory not found:# filepath: /mnt/data1/samk/NEU/mesh-splat/run_animation.sh"
fi