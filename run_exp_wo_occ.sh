
# [DONE] migrate script from DGM

#!/bin/bash
# set -e  # Remove this line to prevent script exit on error
export CUDA_VISIBLE_DEVICES=1

# ====== Simple timers ======
fmt_time() {
    local T=$1
    printf "%02d:%02d:%02d" $((T/3600)) $(((T%3600)/60)) $((T%60))
}
total_start=$(date +%s)
total_exp_seconds=0
failed_experiments=0

# ======= Experiment Parameters ======
BUDGETS=(16384 23170 32768 46341 65536 92682 131072 185364 262144 368589 524288) # Add your budgets here

POLICIES=("planarity" "area" "rand_uni") # Add your policies here
# POLICIES=("texture" "mse_mask")

SCENE_NAME="hotdog" # add a loop for multiple scenes if needed
DATASET_DIR="/mnt/data1/syjintw/NEU/dataset/hotdog"

BASE_OUTPUT_DIR="output/1028_without_occ/${SCENE_NAME}"

# Create the base output directory if it doesn't exist
mkdir -p "$BASE_OUTPUT_DIR"

# Timing summary file
TIMING_SUMMARY="${BASE_OUTPUT_DIR}/timing_summary.tsv"
echo -e "policy\tbudget\tduration_seconds\tduration_hms\tstatus" >> "$TIMING_SUMMARY"

# Failed experiments log
FAILED_LOG="${BASE_OUTPUT_DIR}/failed_experiments.log"
> "$FAILED_LOG"  # Clear the file


# ======= Main Loop ======
for policy in "${POLICIES[@]}"; do
    for budget in "${BUDGETS[@]}"; do
        
        SAVE_DIR="${BASE_OUTPUT_DIR}/${policy}_${budget}/"
        LOG_FILE="log_train_${policy}_${budget}.log"

        # Ensure the save directory exists
        mkdir -p "$SAVE_DIR"

        {
            echo "================================================================="
            echo "Starting experiment: policy=${policy}, budget=${budget}"
            echo "Running on $(hostname), on branch $(git branch --show-current)"
            echo "Read dataset from: $DATASET_DIR"
            echo "Output will be saved to: $SAVE_DIR"
            date +"%Y-%m-%d %H:%M:%S"
            echo "GPU cores: $CUDA_VISIBLE_DEVICES"
            echo "================================================================="
        } | tee "$LOG_FILE"

        exp_start=$(date +%s)

        # Run the experiment and capture the exit status
        if python train.py --eval \
            -s  "$DATASET_DIR" \
            -m "$SAVE_DIR" \
            --texture_obj_path /mnt/data1/syjintw/NEU/dataset/hotdog/mesh.obj \
            --debugging \
            --debug_freq 100 \
            --total_splats "$budget" \
            --alloc_policy "$policy" \
            --gs_type gs_mesh -w --iteration 1000 >> "$LOG_FILE";then
            # no --occlusion



            exp_status="SUCCESS"
            
            # Copy the log file to the save directory
            cp "$LOG_FILE" "$SAVE_DIR/$LOG_FILE"
            echo "Log file saved to $SAVE_DIR/$LOG_FILE"

        else
            exp_status="FAILED"
            failed_experiments=$((failed_experiments + 1))
            echo "ERROR: Experiment failed: policy=${policy}, budget=${budget}" | tee -a "$FAILED_LOG"
            echo "Check log file for details: $LOG_FILE" | tee -a "$FAILED_LOG"
            cp "$LOG_FILE" "$SAVE_DIR/$LOG_FILE" 2>/dev/null || true
        fi

        exp_end=$(date +%s)
        exp_secs=$((exp_end - exp_start))
        exp_hms=$(fmt_time "$exp_secs")
        total_exp_seconds=$((total_exp_seconds + exp_secs))

        {
            echo "Experiment duration: ${exp_hms} (${exp_secs}s)"
            echo "Experiment status: ${exp_status}"
            echo "Finished experiment: policy=${policy}, budget=${budget}"
            echo ""
        } | tee -a "$LOG_FILE"

        printf "%s\t%s\t%d\t%s\t%s\n" "$policy" "$budget" "$exp_secs" "$exp_hms" "$exp_status" >> "$TIMING_SUMMARY"

    done
done



total_end=$(date +%s)
wall_secs=$((total_end - total_start))
echo "================================================================="
echo "All experiments completed."
echo "Wall-clock total: $(fmt_time "$wall_secs") (${wall_secs}s)"
echo "Sum of experiment durations: $(fmt_time "$total_exp_seconds") (${total_exp_seconds}s)"
printf "TOTAL\t\t%d\t%s\tTOTAL_SUM\n" "$total_exp_seconds" "$(fmt_time "$total_exp_seconds")" >> "$TIMING_SUMMARY"
echo "Failed experiments: ${failed_experiments}"
echo "Timing summary saved to: ${TIMING_SUMMARY}"
if [ $failed_experiments -gt 0 ]; then
    echo "Failed experiments log: ${FAILED_LOG}"
fi
echo "================================================================="