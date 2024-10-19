#!/usr/bin/env bash


set -e

# uses: gum, xsel, jack_capture, sox, curl

# Configuration (possibly source from a .env file)
declare -rx timestamp_format='%Y-%m-%d_%H-%M-%S'
# declare -rx dictations="$HOME/Desktop/Dictations"
declare -rx dictations_tmp="/tmp/audio/dictations"

declare -rx localwhisper="http://tinybot.syncopated.net:8082/inference"

if ! /usr/bin/curl -s "$localwhisper" > /dev/null; then
  echo "Error: Whisper service is not running at $localwhisper. Please start it before running this script."
  exit 1
fi

TMPDIR=$(mktemp --directory)

declare -rx tmpmarkdown=$(mktemp -p ${TMPDIR} --suffix .md)

# Get the notebook to record to
echo "choose a notebook\n"
notebook=$(gum file --height=10 --directory ~/Desktop)
dictations="${notebook}/_dictations"

# Function to get the output file path
get_output_file() {
  if [[ -z "$1" ]]; then
	# No file name provided, prompt for one
	output_file=$(gum file --height=10 ~/Desktop)
  else
	# Use the provided file name
	output_file="$1"
  fi
  echo "$output_file"
}

# Capture audio with default settings for Reaper outputs
function capture_audio() {
	capture_args=(
		-s
		-f wav
		-b 16  # Adjust bit depth if needed
		-d 600   # Adjust duration limit if needed
		-c 2    # Capture from Reaper outputs (out1, out2)
		-p 'REAPER:out1' -p 'REAPER:out2'
	)

	tput cup 15 0
	# Execute capture and conversion
	jack_capture "${capture_args[@]}" "${dictations}/${wavfile}" &&
		sox "${dictations}/${wavfile}" -r 16000 -b 16 "${tmpwavfile}" remix 1-2
}

function termput() {
	tput clear && tput cup 15 0
}

function cleanup() {
	fd ".wav" $dictations_tmp -X rm {}
}

# Main Script Logic
trap cleanup SIGINT SIGTERM ERR EXIT

gum style \
	--foreground 014 --border-foreground 024 --border double \
	--align center --width 50 --margin "1 2" --padding "2 4" \
	'Hello.' && sleep 1

if ! [ -d $dictations_tmp ]; then
	mkdir -pv $dictations_tmp
	echo "tmp dictations dir created!"
fi

# Get the output file path
output_file=$(get_output_file "$1")

while true; do

	# Determine jack_capture options and filenames
	declare -x timestamp=$(date +"$timestamp_format")
	declare -x capture_file="jack_capture_${timestamp}"
	declare -x wavfile="${capture_file}.wav"
	declare -x tmpwavfile="${dictations_tmp}/${capture_file}.wav"


	capture_audio

	if [[ $? -ne 0 ]]; then
		notify-send -t 5000 -u critical "Capture initialization failed"
		exit
	else
		tput clear && notify-send -t 5000 'transcribing....'
		logger -t dictation --priority user.debug transcribing ${dictations}/${wavfile}
        # Use gum spin for transcription
        text=$(gum spin -s line --title "Transcribing..." --show-output -- \
            curl $localwhisper \
            -H "Content-Type: multipart/form-data" \
            -F file="@${tmpwavfile}" \
            -F temperature="0.2" | jq -r .text | ruby -pe 'gsub(/^\s/, "")')

		if [[ "$text" == "null" ]]; then
			notify-send -t 5000 -u critical "Transcription failed"
		else
			echo "$text" >> "$output_file"
			notify-send -t 5000 "Transcription saved to: $output_file"
		fi
	fi

	termput

	next=$(gum choose --timeout=6s --selected="continue" "continue" "exit")

	if [[ "${next}" == "exit" ]]; then
		break
	fi
done

termput

cleanup

tput clear && tput cup 20 40 && gum spin --spinner dot --spinner.margin="2 2" --title "exiting in 2 seconds..." -- sleep 2
