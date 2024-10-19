#!/usr/bin/env ruby

require 'fileutils'
require 'tempfile'

# Function to run whisper
def run_whisper(wav_file)
  whisper = ENV['HOME'] + '/Workspace/Vox/whisper.cpp'
  model = whisper + '/models/ggml-base.en-q5_0.bin'

  system("#{whisper}/build/bin/main", '-m', model, '-pp', '-sow',
         '-ojf', '-otxt', '-osrt', '-ocsv', '-ovtt',
         '-of', "#{@dest_dir}/#{@filename}",
         '-pc', wav_file)
end

# Function to run sonic-annotator
def run_sonic_annotator(wav_file)
  system('sonic-annotator', '-d', 'vamp:vamp-aubio:aubiomelenergy:mfcc',
         '-d', 'vamp:vamp-aubio:aubioonset:onsets',
         '-d', 'vamp:vamp-aubio:aubionotes:notes',
         '-d', 'vamp:mtg-melodia:melodia:melody', wav_file,
         '-w', 'csv', '--csv-force', '--csv-basedir', @dest_dir)
end

# Function to process video using ffmpeg
def transcode(infile, outfile)
  crf = '15.0'
  vcodec = 'libx264'
  acodec = 'copy'
  coder = '1'
  me_method = 'hex'
  subq = '6'
  me_range = '16'
  g = '250'
  keyint_min = '25'
  sc_threshold = '40'
  i_qfactor = '0.71'
  b_strategy = '1'
  strict = '-2'
  threads = '19'

  system('ffmpeg', '-i', infile,
         '-crf', crf,
         '-vcodec', vcodec,
         '-acodec', acodec,
         '-coder', coder,
         '-flags', '+loop', '-cmp', '+chroma', '-partitions', '+parti4x4+partp8x8+partb8x8',
         '-me_method', me_method,
         '-subq', subq,
         '-me_range', me_range,
         '-g', g,
         '-keyint_min', keyint_min,
         '-sc_threshold', sc_threshold,
         '-i_qfactor', i_qfactor,
         '-b_strategy', b_strategy,
         '-strict', strict,
         '-threads', threads,
         '-y', outfile)
end

# Function to extract audio from a video file
def extract_audio(infile, outfile, compress = nil)
  system('ffmpeg', '-i', infile, '-ar', '16000', '-acodec', 'pcm_s16le', '-ac', '1', outfile)

  if compress == 'mp3'
    mp3file = "#{@temp_dir}/#{@filename}.mp3"
    system('sox', outfile, '-b', '16', mp3file)
  end
end

# Function to run the unsilence command
def unsilence_audio(infile, outfile)
  threshold = '-30' # Default threshold
  puts 'Enter threshold: (-30)'
  input = gets.chomp
  threshold = input unless input.empty?

  system('unsilence', '-d', '-ss', '1.5', '-sl', threshold, infile, outfile)
end

def normalize(infile, outfile)
  system('ffmpeg-normalize', '-pr', '-nt', 'rms', infile,
         '-prf', 'highpass=f=200', '-prf', 'dynaudnorm=p=0.4:s=15', '-pof', 'lowpass=f=7000',
         '-ar', '48000', '-c:a', 'pcm_s16le', '--keep-loudness-range-target',
         '-o', outfile)
end

def pipeline(infile)
  normalized = "#{@temp_dir}/#{@filename}_normalized.#{@ext}"
  wavfile = "#{@temp_dir}/#{@filename}.wav"
  outfile = "#{@temp_dir}/#{@filename}.#{@ext}"

  normalize(infile, normalized)
  unsilence_audio(normalized, outfile)
  extract_audio(outfile, wavfile)
  run_whisper(wavfile) && run_sonic_annotator(wavfile)
end

# Function to move files to a destination folder
def move_files(source_dir, dest_dir)
  FileUtils.mkdir_p(dest_dir)
  FileUtils.mv(Dir.glob("#{source_dir}/*"), dest_dir)
end

# Main script logic
if ARGV.length < 1
  puts 'Usage: process_video.rb <infile> [<dest_dir>]'
  exit 1
end

@infile = ARGV[0]
@fbasename = File.basename(@infile)
@filename = File.basename(@infile, '.*')
@ext = File.extname(@infile)[1..-1]

# Get file size in bytes
file_size = File.size(@infile)

# Get available RAM in bytes
available_ram = `free -m`.match(/Mem:\s+(\d+)/)[1].to_i * 1024 * 1024

# Determine temp directory based on available RAM
@temp_dir = if available_ram > (file_size * 2)
              Dir.mktmpdir
            else
              Dir.mktmpdir(nil, '/var/tmp')
            end

@dest_dir = if ARGV.length > 1
              ARGV[1]
            else
              puts 'Enter destination directory:'
              gets.chomp
            end

puts 'Choose an action:'
puts '1. Transcode'
puts '2. Extract Audio'
puts '3. Normalize Audio'
puts '4. Unsilence Audio'
puts '5. Pipeline'
choice = gets.chomp.to_i

case choice
when 1
  outfile = "#{@temp_dir}/#{@filename}.mp4"
  transcode(@infile, outfile)
when 2
  outfile = "#{@temp_dir}/#{@filename}.wav"
  extract_audio(@infile, outfile)
when 3
  outfile = "#{@temp_dir}/#{@filename}_normalized.#{@ext}"
  normalize(@infile, outfile)
when 4
  outfile = "#{@temp_dir}/#{@filename}.#{@ext}"
  unsilence_audio(@infile, outfile)
when 5
  pipeline(@infile)
else
  puts 'Invalid choice.'
end

move_files(@temp_dir, @dest_dir)

FileUtils.rm_rf(@temp_dir)

puts 'Done!'
