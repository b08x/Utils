#!/bin/bash

# Check for dependencies
if ! command -v sox &>/dev/null; then
    echo "Error: sox is required for this script. Please install it."
    exit 1
fi

if ! command -v fd &>/dev/null; then
    echo "Error: fd command is required for this script. Please install it."
    exit 1
fi

sox /tmp/output.wav -r 16000 -b 16 -c 1 snotes_mono16kv23.wav highpass 220 vol 1.5 amplitude 0.3 \\nsilence 0 0.1 00:00:00.1 -60d : newfile : restart

# Set the path to search for audio files
path="/tmp"

# Use fd to find audio files (adjust extensions as needed)
fd --extension mp3 --extension wav --extension flac -p "$path" | while read -r file; do
    # Run sox and capture the output
    output=$(sox -V2 "$file" -n stats 2>&1)

    # Check if the output indicates no audio
    if [[ "$output" == "sox WARN stats: no audio" ]]; then
        echo "Deleting file with no audio: $file"
        rm "$file"
    fi
done

