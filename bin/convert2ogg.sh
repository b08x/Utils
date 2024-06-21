#!/bin/bash

# Get the input file name
INPUT_FILE=$1

# Get the file extension of the input file
EXTENSION=${INPUT_FILE##*.}

# Normalize the audio using ffmpeg-normalize
ffmpeg-normalize "$INPUT_FILE" -o "${INPUT_FILE%.*}-normalized.${EXTENSION}" 

# Convert the normalized file to OGG
ffmpeg -i "${INPUT_FILE%.*}-normalized.${EXTENSION}" -c:a libvorbis -b:a 192k -ar 44100 "${INPUT_FILE%.*}.ogg"