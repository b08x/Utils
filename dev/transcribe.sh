#!/usr/bin/env bash
set -x

declare -rx timestamp=$(date +'%Y-%m-%d_%H-%M-%S')

declare -rx dictations="$HOME/Recordings/audio/dictations"

# declare -rx transcription="/tmp/transcription-${timestamp}.txt"

declare -rx capturefile="${dictations}/jack_capture_${timestamp}"

declare -rx wavfile="${capturefile}.wav"

declare xel_args="-ib"

function cleanup() {
    cd /tmp && rm -rf *.mp3
}

function transcribe() {
        local infile=$1
        fbasename=$(basename "$1")
        filename="${fbasename%.*}"

        local outfile="/tmp/${filename}.mp3"

        sox ${infile} ${outfile}

        text=$(gum spin -s line --title "transcribing" --show-output -- curl https://api-inference.huggingface.co/models/distil-whisper/distil-large-v2 \
                        -X POST \
                        --data-binary "@${outfile}" \
                        -H "Authorization: Bearer $HUGGINGFACE_API_KEY"
                      )

        echo $text | jq '.text' | xargs -0 | ruby -pe 'gsub(/^\s/, "")'
}

#########################################################################
#                             Greetings                                 #
#########################################################################
trap cleanup SIGINT SIGTERM ERR EXIT

gum style \
        --foreground 014 --border-foreground 024 --border double \
        --align center --width 50 --margin "1 2" --padding "2 4" \
        'Hello.' && sleep 1

xsel -cb

# transcribe $1

if [ $? = 0 ]; then
        notify-send -t 5000 'transcribing....'
        logger -t dictation --priority user.debug transcribing $1

        text=""
        max_retries=5
        for ((i=0; i<max_retries; i++)); do
                text=$(transcribe $1)
                if [[ "$text" != "null" ]]; then
                        break
                fi

                echo $text

                echo "Transcription failed, retrying ($((max_retries-i)))..."

                gum spin -s line --title "waiting 10 seconds..." sleep 10

        done

        if [[ "$text" == "null" || "$text" == "" ]]; then
                notify-send -t 5000 -u critical "Transcription failed"
        else
                text=$(echo $text | xargs -0 | ruby -pe 'gsub(/^\s/, "")')
                xsel -a -b <<< "$text"
                notify-send -t 10000 "transcription copied to clipboard"
                xsel -ob | gum format -t code
        fi

else
        notify-send -t 5000 -u critical "capture failed"
fi
