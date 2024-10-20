#!/usr/bin/env bash
set -x

# Create a temporary directory in /tmp
temp_dir=$(mktemp -d -p /tmp)

# Function to run whisper
run_whisper() {
  local wav_file="$1"

  WHISPER="$HOME/Workspace/Vox/whisper.cpp"
  MODEL="$WHISPER/models/ggml-base.en-q5_0.bin"
  # -owts -fp /usr/local/share/fonts/Hack/Hack\ Bold\ Nerd\ Font\ Complete.ttf

  $WHISPER/build/bin/main -m $MODEL -pp -sow \
    -ojf -otxt -osrt -ocsv -ovtt \
    -of "${dest_dir}/${filename}" \
    -pc "${wav_file}"
}

# Function to run sonic-annotator
run_sonic_annotator() {
  local wav_file="$1"

  sonic-annotator -d vamp:vamp-aubio:aubiomelenergy:mfcc \
    -d vamp:vamp-aubio:aubioonset:onsets \
    -d vamp:vamp-aubio:aubionotes:notes \
    -d vamp:mtg-melodia:melodia:melody "${wav_file}" \
    -w csv --csv-force --csv-basedir "${dest_dir}"
}

# Function to process video using ffmpeg
transcode() {
  local infile="$1"
  local outfile="$2"

  # ffmpeg options
  local crf="15.0"
  local vcodec="libx264"
  local acodec="copy"
  local coder="1"
  local me_method="hex"
  local subq="6"
  local me_range="16"
  local g="250"
  local keyint_min="25"
  local sc_threshold="40"
  local i_qfactor="0.71"
  local b_strategy="1"
  local strict="-2"
  local threads="19"

  /usr/bin/ffmpeg -i "${infile}" \
    -crf "${crf}" \
    -vcodec "${vcodec}" \
    -acodec "${acodec}" \
    -coder "${coder}" \
    -flags +loop -cmp +chroma -partitions +parti4x4+partp8x8+partb8x8 \
    -me_method "${me_method}" \
    -subq "${subq}" \
    -me_range "${me_range}" \
    -g "${g}" \
    -keyint_min "${keyint_min}" \
    -sc_threshold "${sc_threshold}" \
    -i_qfactor "${i_qfactor}" \
    -b_strategy "${b_strategy}" \
    -strict "${strict}" \
    -threads "${threads}" \
    -y "${outfile}"
}

# Function to extract audio from a video file
extract_audio() {
  local infile="$1"
  local outfile="$2"
  local compress="$3"
  # Extract audio using ffmpeg
  ffmpeg -i "${infile}" -ar 16000 -acodec pcm_s16le -ac 1 "${outfile}"

  if [[ "$compress" == "mp3" ]]; then
    mp3file="${temp_dir}/${filename}.mp3"
    sox "${outfile}" -b 16"${mp3file}.mp3"
  fi

}

# Function to run the unsilence command
unsilence_audio() {
  local infile="$1"
  local outfile="$2"
  local threshold=$(gum input --prompt "Enter threshold: " --placeholder="-30" --value="-30")

  unsilence -d -ss 1.5 -sl "${threshold}" "${infile}" "${outfile}"

}

normalize() {
  local infile="$1"

  ffmpeg-normalize -pr -nt rms "${infile}" \
    -prf "highpass=f=200" -prf "dynaudnorm=p=0.4:s=15" -pof "lowpass=f=7000" \
    -ar 48000 -c:a pcm_s16le --keep-loudness-range-target \
    -o "${normalized}"
}

pipeline() {
  local infile="$1"
  local ext="${infile##*.}"
  local normalized="${temp_dir}/${filename}_normalized.${ext}"
  local wavfile="${temp_dir}/${filename}.wav"

  local outfile="${temp_dir}/${filename}.${ext}"

  normalize "${infile}" "${normalized}"

  unsilence_audio "${normalized}" "${outfile}"

  extract_audio "${outfile}" "${wavfile}"

  run_whisper "${wavfile}" && run_sonic_annotator "${wavfile}" "${dest_dir}"

}

# Function to move files to a destination folder
move_files() {
  local source_dir="$1"
  local dest_dir="$2"

  # Create the destination folder if it doesn't exist
  mkdir -p "${dest_dir}"

  # Move all files from the source directory to the destination directory
  mv "${source_dir}"/* "${dest_dir}"
}

# Main script logic
declare infile="$1"
declare fbasename=$(basename "$infile")
declare filename="${fbasename%.*}"
declare ext="${infile##*.}"

# Get file size in bytes
file_size=$(stat -c%s "$infile")

# Get available RAM in bytes
available_ram=$(free -m | awk '/Mem:/ {print $4}')
available_ram=$((available_ram * 1024 * 1024))

# Determine temp directory based on available RAM
if ((available_ram > (file_size * 2))); then
  temp_dir=$(mktemp -d -p /tmp)
else
  temp_dir=$(mktemp -d -p /var/tmp)
  export TMPDIR="/var/tmp"
fi

if [[ -d "$2" ]]; then
  dest_dir="$2"
else
  dest_dir=$(yad --file --directory)
fi

CHOICE=$(gum choose "Transcode" "Extract Audio" "Normalize Audio" "Unsilence Audio" "Pipeline")

case "$CHOICE" in
"Pipeline")
  pipeline "${infile}"
  ;;
"Transcode")
  outfile="${temp_dir}/${filename}.mp4"
  transcode "${infile}" "${outfile}"
  ;;
"Extract Audio")
  outfile="${temp_dir}/${filename}"
  extract_audio "${infile}" "${outfile}.wav"
  ;;
"Normalize Audio")
  outfile="${temp_dir}/${filename}"
  normalize "${infile}" "${outfile}_normalized.${ext}"
  ;;
"Unsilence Audio")
  outfile="${temp_dir}/${filename}"
  unsilence_audio "${infile}" "${outfile}"
  ;;
esac

move_files "${temp_dir}" "${dest_dir}"

rm -rf "${temp_dir}"

# # Ask the user if they want to move the files to a permanent folder
# MOVE_CHOICE=$(gum choose "Move to permanent folder" "Keep in temporary folder")

# if [[ "$MOVE_CHOICE" == "Move to permanent folder" ]]; then
#   # Ask for the destination folder
#   dest_dir=$(yad --file --directory)

#   # Check if the user canceled the folder selection
#   if [[ $? -ne 0 ]]; then
#     echo "Folder selection canceled. Files will remain in the temporary directory."
#   else
#     # Move the files to the destination folder
#     move_files "${temp_dir}" "${dest_dir}"

#     # Ask the user if they want to remove the temporary directory
#     rm_tmp=$(gum choose --timeout=6s --selected="yes" "yes" "no")
#     if [[ "$rm_tmp" == "yes" ]]; then
#       rm -rf "${temp_dir}"
#     else
#       echo "Files will remain in the temporary directory."
#     fi

#   fi
# fi
