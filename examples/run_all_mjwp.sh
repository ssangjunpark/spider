#!/bin/bash

# Define constants
DATASET_NAME="lafan"
ROBOT_TYPE="unitree_g1"
HAND_TYPE="humanoid"
DATA_ID=0

# Define the base directory where tasks are located
BASE_DIR="example_datasets/processed/${DATASET_NAME}/${ROBOT_TYPE}/${HAND_TYPE}"

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
  echo "Error: Directory $BASE_DIR does not exist."
  exit 1
fi

TOTAL_START_TIME=$(date +%s)

# Iterate over each subdirectory in the base directory
for TASK_DIR in "$BASE_DIR"/*; do
  if [ -d "$TASK_DIR" ]; then
    # Extract the task name from the directory path
    TASK=$(basename "$TASK_DIR")

    TASK_START_TIME=$(date +%s)
    echo "Processing task: $TASK"

    # Run the command
    # Using 'conda run' (assuming 'spider' environment)
    # Added viewer=none to disable popup window
    # Added save_video=true to ensure video rendering is kept
    # Added --no-capture-output to allow printing to stdout
    conda run --no-capture-output -n spider python -u examples/run_mjwp.py \
      +override=humanoid \
      dataset_name="${DATASET_NAME}" \
      task="${TASK}" \
      data_id="${DATA_ID}" \
      robot_type="${ROBOT_TYPE}" \
      embodiment_type="${HAND_TYPE}" \
      viewer=none \
      save_video=true

    # Check exit status
    EXIT_STATUS=$?
    TASK_END_TIME=$(date +%s)
    TASK_DURATION=$((TASK_END_TIME - TASK_START_TIME))

    if [ $EXIT_STATUS -eq 0 ]; then
      echo "Successfully processed task: $TASK"
    else
      echo "Failed to process task: $TASK"
    fi
    echo "Task duration: ${TASK_DURATION} seconds"

    echo "---------------------------------------------------"
  fi
done

TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
echo "All tasks processed."
echo "Total duration: ${TOTAL_DURATION} seconds"
