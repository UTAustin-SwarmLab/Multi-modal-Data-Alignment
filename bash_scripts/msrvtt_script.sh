#!/bin/bash

# Replace this with your command
COMMAND="poetry run python mmda/any2any_conformal_retrieval.py dataset=MSRVTT"

# Function to start the process
start_process() {
    $COMMAND &
    PID=$!
}

# Start the process for the first time
start_process

# Initialize variables for exponential backoff
WAIT_TIME=5  # Initial wait time (in seconds)
MAX_WAIT_TIME=3600  # Maximum wait time (1 hour)
# Monitor the process and restart if it dies
while true; do
    if ! ps -p $PID > /dev/null; then
        echo "$(date): Process $PID has been killed. Restarting..." >> $LOGFILE
        start_process

        # Exponentially increase the wait time
        echo "$(date): Waiting for $WAIT_TIME seconds before checking again..." >> $LOGFILE
        sleep $WAIT_TIME

        # Increase the wait time exponentially, but cap it at MAX_WAIT_TIME
        WAIT_TIME=$((WAIT_TIME * 2))
        if [ $WAIT_TIME -gt $MAX_WAIT_TIME ]; then
            WAIT_TIME=$MAX_WAIT_TIME
        fi
    else
        # Reset the wait time if the process is running normally
        WAIT_TIME=5
    fi
done
