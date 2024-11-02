#!/usr/bin/env python3

import os
import shutil
import re
from pathlib import Path
import argparse
import urllib.parse

def get_asset_subfolder(file_path):
    """Determine the appropriate assets subfolder based on file extension"""
    extension = file_path.lower().split('.')[-1]
    
    if extension in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'svg', 'webp']:
        return 'img'
    elif extension in ['mp3', 'wav', 'ogg', 'm4a', 'flac']:
        return 'audio'
    elif extension in ['mp4', 'mov', 'avi', 'mkv', 'webm']:
        return 'video'
    elif extension == 'pdf':
        return 'pdf'
    else:
        return 'other'

def extract_title_from_path(path):
    """Extract the title from a markdown file path"""
    # Remove the .md extension if present
    base_name = os.path.basename(path)
    if base_name.lower().endswith('.md'):
        base_name = base_name[:-3]
    
    # URL decode the name
    decoded_name = urllib.parse.unquote(base_name)
    return decoded_name

def copy_markdown_files(source, destination_folder):
    source = os.path.expanduser(source)
    destination_folder = os.path.expanduser(destination_folder)
    
    if os.path.isdir(source):
        base_folder_name = os.path.basename(os.path.normpath(source))
        notes_folder = os.path.join(destination_folder, base_folder_name)
    else:
        notes_folder = destination_folder
    
    os.makedirs(notes_folder, exist_ok=True)
    
    root_destination = os.path.dirname(destination_folder.rstrip(os.path.sep))
    assets_root = os.path.join(root_destination, 'assets')
    
    asset_subfolders = ['img', 'audio', 'video', 'pdf', 'other']
    for subfolder in asset_subfolders:
        os.makedirs(os.path.join(assets_root, subfolder), exist_ok=True)
    
    def process_file(file_path):
        source_folder = os.path.dirname(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Process existing double-bracketed links first
        backlinks = re.findall(r'\[\[(.*?)\]\]', content)
        for backlink in backlinks:
            backlink_path = os.path.join(source_folder, f"{backlink}.md")
            if os.path.exists(backlink_path):
                print(f"Processing backlink: {backlink_path}")
                process_file(backlink_path)
        
        # Convert markdown links to double brackets if they point to .md files
        def convert_markdown_link(match):
            text = match.group(1)
            link = match.group(2)
            
            # Check if it's a markdown file link
            if link.lower().endswith('.md'):
                # Extract the title from the path
                title = extract_title_from_path(link)
                # Process the linked file if it exists
                full_link_path = os.path.join(source_folder, link)
                if os.path.exists(full_link_path):
                    print(f"Processing markdown link: {full_link_path}")
                    process_file(full_link_path)
                return f"[[{title}]]"
            
            # Return original link if it's not a markdown file
            return match.group(0)
        
        # Convert markdown links to double brackets
        content = re.sub(r'\[(.*?)\]\((.*?)\)', convert_markdown_link, content)
        
        # Process attachments and wrap images with liquid tags
        def replace_attachment(match):
            alt_text, attachment_path = match.groups()
            full_attachment_path = os.path.join(source_folder, attachment_path)
            
            if os.path.exists(full_attachment_path):
                asset_subfolder = get_asset_subfolder(attachment_path)
                new_attachment_name = os.path.basename(attachment_path)
                new_attachment_path = f'/assets/{asset_subfolder}/{new_attachment_name}'
                dest_asset_path = os.path.join(root_destination, new_attachment_path.lstrip('/'))
                shutil.copy2(full_attachment_path, dest_asset_path)
                
                if asset_subfolder == 'img':
                    return f'{{% picture {new_attachment_name} --alt {alt_text} %}}'
                else:
                    return f'![{alt_text}]({new_attachment_path})'
            return match.group(0)

        # Only process image/attachment links (not already processed markdown links)
        content = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_attachment, content)
        
        # Process Mermaid and PlantUML code blocks
        def process_code_blocks(match):
            code_type = match.group(1).lower()
            code_content = match.group(2)
            
            if code_type == 'mermaid':
                return f'```mermaid!\n{code_content}\n```\n\nMermaid diagram detected. Consider rendering this diagram.'
            elif code_type == 'plantuml':
                return f'```plantuml!\n{code_content}\n```\n\nPlantUML diagram detected. Consider rendering this diagram.'
            else:
                return match.group(0)

        content = re.sub(r'```(\w+)\n(.*?)```', process_code_blocks, content, flags=re.DOTALL)
        
        if os.path.isdir(source):
            rel_path = os.path.relpath(file_path, source)
            dest_path = os.path.join(notes_folder, rel_path)
        else:
            dest_path = os.path.join(notes_folder, os.path.basename(file_path))
            
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        with open(dest_path, 'w', encoding='utf-8') as file:
            file.write(content)
    
    def list_markdown_files(folder):
        markdown_files = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.md'):
                    markdown_files.append(os.path.join(root, file))
        return markdown_files
    
    if os.path.isfile(source):
        process_file(source)
    elif os.path.isdir(source):
        markdown_files = list_markdown_files(source)
        for file_path in markdown_files:
            process_file(file_path)
    elif isinstance(source, list):
        for file_path in source:
            if os.path.isfile(file_path) and file_path.endswith('.md'):
                process_file(file_path)
    else:
        raise ValueError("Invalid source. Must be a file, folder, or list of files.")

def main():
    parser = argparse.ArgumentParser(description="Copy Markdown files with double bracket link conversion.")
    parser.add_argument("source", nargs='+', help="Source file(s) or folder")
    parser.add_argument("destination", help="Destination folder")
    args = parser.parse_args()

    source = args.source[0] if len(args.source) == 1 else args.source
    copy_markdown_files(source, args.destination)

if __name__ == "__main__":
    main()
