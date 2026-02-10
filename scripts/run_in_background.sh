#!/bin/bash

# wrapper to run python training scripts in background using nohup

if [ -z "$1" ]; then
    echo "Usage: source scripts/run_in_background.sh <script_path>"
    echo "Example: source scripts/run_in_background.sh scripts/train_unet.py"
    return 1 2>/dev/null || exit 1
fi

SCRIPT=$1
SCRIPT_NAME=$(basename "$SCRIPT" .py)
DATE=$(date +%Y%m%d-%H%M)
LOG_FILE="${SCRIPT_NAME}_${DATE}_nohup.log"

echo "----------------------------------------------------------------"
echo "Starting $SCRIPT in background..."
echo "Console output (stdout/stderr) will be saved to: $LOG_FILE"
echo "Training logs/checkpoints will be saved to: models/$SCRIPT_NAME-$DATE/ (managed by the python script)"
echo "----------------------------------------------------------------"

nohup python "$SCRIPT" > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process ID: $PID"
echo "To check status: ps -p $PID"
echo "To kill process: kill $PID"
echo "To watch output: tail -f $LOG_FILE"
echo "----------------------------------------------------------------"
