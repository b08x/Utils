#!/usr/bin/env ruby
# frozen_string_literal: true

require 'fileutils'

def create_folder_structure(base_path)
  structure = {
    'Text_Generation' => ['General_Expansion', 'Topic_Specific', 'Creative_Writing'],
    'Formatting' => ['Punctuation_Correction', 'Structure_Modification', 'Style_Adjustment'],
    'Image_Generation' => ['Descriptive', 'Style-based', 'Scene_Composition'],
    'System_Instructions' => ['DevOps_Engineer', 'Data_Scientist', 'Creative_Writer', 'Customer_Support', 'Other_Specialized_Roles'],
    'Text_Merging' => ['Information_Synthesis', 'Comparative_Analysis', 'Multi-source_Integration'],
    'Templates' => ['Titles', 'Tags', 'Headings', 'Product_Descriptions'],
    'Perspective_Shifting' => ['Alternative_Viewpoints', 'Contrarian_Angles', 'Empathy_Building'],
    'Meta' => ['Best_Practices', 'Prompt_Engineering_Tips', 'Version_History'],
    'Custom' => []
  }

  structure.each do |folder, subfolders|
    folder_path = File.join(base_path, folder)
    FileUtils.mkdir_p(folder_path)
    puts "Created folder: #{folder_path}"

    subfolders.each do |subfolder|
      subfolder_path = File.join(folder_path, subfolder)
      FileUtils.mkdir_p(subfolder_path)
      puts "Created subfolder: #{subfolder_path}"
    end
  end
end

def get_base_directory
  if ARGV.empty?
    print "Enter the base directory for the Prompt Library: "
    gets.chomp.strip
  else
    ARGV[0]
  end
end

def process_directory(base_directory)
  # Remove surrounding quotes if present
  base_directory = base_directory.gsub(/\A["']|["']\Z/, '')

  # Expand the path to handle '~' for home directory
  base_directory = File.expand_path(base_directory)

  # Check if the directory exists
  unless Dir.exist?(base_directory)
    puts "The specified directory does not exist. Would you like to create it? (y/n)"
    response = gets.chomp.downcase
    if response == 'y'
      FileUtils.mkdir_p(base_directory)
      puts "Created directory: #{base_directory}"
    else
      puts "Exiting without creating the folder structure."
      exit
    end
  end

  base_directory
end

# Get and process the base directory
base_directory = process_directory(get_base_directory)

# Create the folder structure
create_folder_structure(base_directory)

puts "Folder structure created successfully!"
