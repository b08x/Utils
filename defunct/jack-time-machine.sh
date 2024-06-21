# Set your desired options
BITDEPTH="FLOAT"
CHANNELS=1
PORT="handheld:capture_1"
FILENAME_PREFIX="jack_capture_"
FORMAT="wav"
TIMEMACHINE_PREBUFFER=10

# Specify the output directory with a timestamp
OUTPUT_DIRECTORY="$HOME/Recordings/audio/dictations"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Function to stop recording via OSC
stop_recording() {
    oscsend localhost 7777 /jack_capture/tm/stop
}

cleanup() {
  echo "all set!"
  # Stop recording via OSC
  stop_recording
  # if [[ ! $DEBUG ]]; then
  #   unmount_chroot
  #   # rm -rf $SRCDEST && mkdir -p $SRCDEST
  # fi
}

trap cleanup SIGINT SIGTERM ERR EXIT
# Start jack_capture as a daemon with specified options and output directory
jack_capture --bitdepth $BITDEPTH --channels $CHANNELS  --filename-prefix $FILENAME_PREFIX --format $FORMAT --port $PORT --timemachine --timemachine-prebuffer $TIMEMACHINE_PREBUFFER --filename "$OUTPUT_DIRECTORY/$FILENAME_PREFIX$TIMESTAMP.$FORMAT" --osc 7777 --daemon &


# Get the process ID of the jack_capture daemon
PID=$!

# Wait for user input to stop recording (you can customize this part)
# read -p "Press Enter to stop recording..."

# Wait for jack_capture to finish and close gracefully
wait $PID
