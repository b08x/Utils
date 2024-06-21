#!/usr/bin/env ruby
# frozen_string_literal: true

# #! /bin/bash
# ffmpeg -i Long_Audio_File.mp3 -af silencedetect=d=0.1 -f null - |& awk '/silence_end/ {print $4,$5}' | awk '{S=$2;printf "%d:%02d:%02d\n",S/(60*60),S%(60*60)/60,S%60}'
#
# #! /bin/bash
# x="00:00:00"
# z=0
# filename=$(basename -- "$2")
# ext="${filename##*.}"
# filename="${filename%.*}"
# initcmd="ffmpeg  -nostdin -hide_banner -loglevel error -i $2"
# while read y
# do
# initcmd+=" -ss $x -to $y -c copy $filename$z.$ext"
# let "z=z+1"
# x=$y
# done < $1
# $initcmd



require 'open3'
require 'fileutils'

filename = ARGV[1]
ext = File.extname(filename)
basename = File.basename(filename, ext)

# Run the ffmpeg command to detect silence and parse the output
output, _error, _status = Open3.capture3(
  "ffmpeg -i #{filename} -af silencedetect=d=0.1 -f null -"
)
timechunks = output.scan(/silence_end: ([\d\.]+)/).map { |match| match[0] }

# Convert the timechunks into HH:MM:SS format
timechunks.map! do |chunk|
  seconds = chunk.to_i
  hours = seconds / (60 * 60)
  seconds %= (60 * 60)
  minutes = seconds / 60
  seconds %= 60
  sprintf("%02d:%02d:%02d", hours, minutes, seconds)
end

# Process the chunks
initcmd = "ffmpeg -nostdin -hide_banner -loglevel error -i #{filename}"
timechunks.each_with_index do |chunk, index|
  initcmd += " -ss #{index.zero? ? '00:00:00' : timechunks[index-1]} -to #{chunk} -c copy #{basename}#{index}#{ext}"
end

# Run the final ffmpeg command
system(initcmd)
