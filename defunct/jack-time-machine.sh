#!/usr/bin/env bash

# Set your desired options
BITDEPTH="16"
RATE=16000
CHANNELS=1
PORT="REAPER:out"
FILENAME_PREFIX="jack_capture_"
FORMAT="wav"
TIMEMACHINE_PREBUFFER=10

# Specify the output directory with a timestamp
OUTPUT_DIRECTORY="$HOME/Desktop/Dictations"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Function to stop recording via OSC
stop_recording() {
    oscsend localhost 7777 /jack_capture/tm/stop
}

cleanup() {
  echo "all set!"
  # Stop recording via OSC
  stop_recording
}

# Function to handle XF86AudioRecord events
handle_record_key() {
  # Get current recording state from jack_capture (you might need to adjust this command)
  # For example, you could use `jack_lsp` to check if the client is running
  is_recording=$(jack_lsp | grep jack_capture)

  if [[ -n "$is_recording" ]]; then
    # Recording is running, so pause or stop it
    echo "XF86AudioRecord key pressed. Pausing/Stopping recording..."
    stop_recording  # Call your existing stop_recording function
  else
    echo "XF86AudioRecord key pressed, but recording is not running."
  fi
}



trap cleanup SIGINT SIGTERM ERR EXIT
# Start jack_capture as a daemon with specified options and output directory
jack_capture --bitdepth $BITDEPTH --channels $CHANNELS  --filename-prefix $FILENAME_PREFIX --format $FORMAT --port "${PORT}1" --port "${PORT}2" --timemachine --timemachine-prebuffer $TIMEMACHINE_PREBUFFER --filename "$OUTPUT_DIRECTORY/$FILENAME_PREFIX$TIMESTAMP.$FORMAT" --osc 7777 --daemon &


# Get the process ID of the jack_capture daemon
PID=$!

# Wait for user input to stop recording (you can customize this part)
# Use `xev` to determine the keycode for XF86AudioRecord
# Run `xev` in a terminal and press the XF86AudioRecord key.
# Note the keycode from the output (e.g., 173).
record_keycode=175  # Replace with the actual keycode

# Set up an event listener using `xinput`
xinput test-xi2 --root | while read -r line; do
  if [[ "$line" =~ "keycode\ $record_keycode\ " ]]; then
    handle_record_key
  fi
done &

# Wait for jack_capture to finish and close gracefully
wait $PID
