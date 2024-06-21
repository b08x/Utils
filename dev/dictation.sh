#!/usr/bin/env bash
set -e

# uses: gum, xsel, xclip, jack_capture, sox, curl, fd, sd

# Configuration (possibly source from a .env file)
declare -rx timestamp_format='%Y-%m-%d_%H-%M-%S'
declare -rx dictations="$HOME/Recordings/audio/dictations"
declare -rx dictations_tmp="/tmp/audio/dictations"


declare -rx localwhisper="http://tinybot.syncopated.net:8081/inference"
declare -rx hfwhisper="https://api-inference.huggingface.co/models/distil-whisper/distil-large-v2"


TMPDIR=$(mktemp --directory)

declare -rx tmpmarkdown=$(mktemp -p ${TMPDIR} --suffix .md)

xsel -cb

declare -rx note="${1}"




function termput() {
	tput clear && tput cup 15 0
}

function obsidian() {

	xsel -ob >> $note

}

function writer() {
	gum write --width=100 --height=30 --char-limit=0 --header.margin="1 1" \
					  --placeholder "Words" --value="$(xsel -ob)" >> $tmpmarkdown

	printf "\n" >> $tmpmarkdown
}

function clippy() {
	# clear the clipboard selection
	xsel -cb
	# then append the current transcription
	xsel -a -b <<<$text

	tput clear &&	obsidian
}

function transcribe_hf() {
  local file="$1"
  text=$(gum spin -s line --title "transcribing (HuggingFace)" --show-output -- /usr/bin/curl $hfwhisper \
    -H "Authorization: Bearer ${HUGGINGFACEHUB_API_TOKEN}" \
    -H "Content-Type: multipart/form-data" \
    -F file="@${file}" \
    -F temperature="0.2"
  )
  echo "$text" | jq -r .text | ruby -pe 'gsub(/^\s/, "")'
}

function transcribe_local() {
  local file="$1"
  text=$(gum spin -s line --title "transcribing (Local Whisper)" --show-output -- /usr/bin/curl $localwhisper \
    -H "Content-Type: multipart/form-data" \
    -F file="@${file}" \
    -F temperature="0.2"
  )
  echo "$text" | jq -r .text | ruby -pe 'gsub(/^\s/, "")'
}

function localwhisper_available() {
  # Increase timeout value (e.g., 2) for more robust checking
  curl --head "http://tinybot.syncopated.net:8081" >/dev/null 2>&1 && return 0
  return 1
}

function capture_audio() {
	capture_args=(
		-s
		-f wav
		-b 24
		-d 600
	)

	if [[ "$mic_connected" == "connected" ]]; then
		capture_args+=(-c 3 -p 'handheld:capture_1' -p 'system:capture_*')
		remix_channels="1-3"
	else
		capture_args+=(-c 2 -p 'system:capture_*')
		remix_channels="1-2"
	fi

	tput cup 15 0
	# Execute capture and conversion
	jack_capture "${capture_args[@]}" "${dictations}/${wavfile}" &&
		if localwhisper_available; then
			sox "${dictations}/${wavfile}" -r 16000 -b 16 "${tmpwavfile}" remix $remix_channels
		else
			sox "${dictations}/${wavfile}" -r 16000 -b 16 "${mp3file}" remix $remix_channels
		fi
}

function load_mic() {
	card=$(cat /proc/asound/cards | grep -E -m1 "PowerMic" | choose 0)
	
	jack_load 'handheld' zalsa_in -i "-d hw:${card},0"
	
	if [[ $? -eq 0 ]]; then
		mic_connected="connected"
	fi
}

function check_mic() {
	if [[ -z $(jack_lsp | grep handheld) ]]; then
		mic_connected="not connected" 
	else
		mic_connected="connected"
	fi
}

function cleanup() {
	fd ".wav|.mp3" $dictations_tmp -X rm {}
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

while true; do

	# if /tmp reads above 1G, cleanup the mp3 & wav files
	if [[ $(/usr/bin/du -ms /tmp 2&>/dev/null | choose 0) -ge 1024 ]]; then
		cleanup
	fi

	# check if the Powermic Handheld is connected and loaded
	check_mic

	# if not, attempt to load it
	if [[ "${mic_connected}" == "not connected" ]]; then
		load_mic || notify-send -t 5000 -u critical "mic not connected"
	fi

	# Determine jack_capture options based on mic presence
	declare -x timestamp=$(date +"$timestamp_format")
	declare -x capture_file="jack_capture_${timestamp}"
	declare -x wavfile="${capture_file}.wav"
	declare -x mp3file="${dictations_tmp}/${capture_file}.mp3"
	declare -x tmpwavfile="${dictations_tmp}/${capture_file}.wav"


	capture_audio

	if [[ $? -ne 0 ]]; then
		notify-send -t 5000 -u critical "Capture initialization failed"
		exit
	else
		tput clear && notify-send -t 5000 'transcribing....'
		logger -t dictation --priority user.debug transcribing ${dictations}/${wavfile}

		text=""
		max_retries=5
		
		termput
		
		for ((i = 0; i < max_retries; i++)); do
			
			if localwhisper_available; then
				text=$(transcribe_local $tmpwavfile)
			else
				text=$(transcribe_hf $mp3file)
			fi
			
			if [[ "$text" != "null" ]]; then
				break
			fi
			echo "Transcription failed, retrying ($((max_retries - i)))..."

			gum spin -s line --title "waiting 10 seconds..." sleep 10
		done

		if [[ "$text" == "null" ]]; then
			notify-send -t 5000 -u critical "Transcription failed"
		else
			clippy && notify-send -t 10000 "transcription copied to clipboard"
		fi
	fi

	termput

	next=$(gum choose --timeout=6s --selected="continue" "continue" "edit" "exit")

	if [[ "${next}" == "exit" ]]; then
		break
	fi

	if [[ "${next}" == "edit" ]]; then
		micro $tmpmarkdown
	fi
	
done

termput

cat $tmpmarkdown | tee >(xsel -i -b) | gum pager --soft-wrap --foreground="186"

tput clear && tput cup 20 40 && gum spin --spinner dot --spinner.margin="2 2" --title "exiting in 2 seconds..." -- sleep 2
