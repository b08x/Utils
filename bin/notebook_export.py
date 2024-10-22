import os
import shutil
import re
from pathlib import Path
import argparse

def get_asset_subfolder(file_path):
    """Determine the appropriate assets subfolder based on file extension"""
    extension = file_path.lower().split('.')[-1]
    
    # Image files
    if extension in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'svg', 'webp']:
        return 'img'
    # Audio files
    elif extension in ['mp3', 'wav', 'ogg', 'm4a', 'flac']:
        return 'audio'
    # Video files
    elif extension in ['mp4', 'mov', 'avi', 'mkv', 'webm']:
        return 'video'
    # PDF files
    elif extension == 'pdf':
        return 'pdf'
    # Default case
    else:
        return 'other'

def copy_markdown_files(source, destination_folder):
    # Resolve home directory if present in paths
    source = os.path.expanduser(source)
    destination_folder = os.path.expanduser(destination_folder)
    
    # If source is a folder, create the same folder name in destination
    if os.path.isdir(source):
        base_folder_name = os.path.basename(os.path.normpath(source))
        notes_folder = os.path.join(destination_folder, base_folder_name)
    else:
        notes_folder = destination_folder
    
    # Create destination folder structure
    os.makedirs(notes_folder, exist_ok=True)
    
    # Create assets folder and subfolders in the root of destination
    root_destination = os.path.dirname(destination_folder.rstrip(os.path.sep))
    assets_root = os.path.join(root_destination, 'assets')
    
    # Create all asset subfolders
    asset_subfolders = ['img', 'audio', 'video', 'pdf', 'other']
    for subfolder in asset_subfolders:
        os.makedirs(os.path.join(assets_root, subfolder), exist_ok=True)
    
    def process_file(file_path):
        source_folder = os.path.dirname(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Process backlinks
        backlinks = re.findall(r'\[\[(.*?)\]\]', content)
        for backlink in backlinks:
            backlink_path = os.path.join(source_folder, f"{backlink}.md")
            if os.path.exists(backlink_path):
                process_file(backlink_path)
        
        # Process attachments and wrap images with liquid tags
        def replace_attachment(match):
            alt_text, attachment_path = match.groups()
            full_attachment_path = os.path.join(source_folder, attachment_path)
            
            if os.path.exists(full_attachment_path):
                # Determine appropriate subfolder
                asset_subfolder = get_asset_subfolder(attachment_path)
                
                # Create new path within appropriate subfolder
                new_attachment_name = os.path.basename(attachment_path)
                new_attachment_path = f'/assets/{asset_subfolder}/{new_attachment_name}'
                
                # Copy file to appropriate subfolder
                dest_asset_path = os.path.join(root_destination, new_attachment_path.lstrip('/'))
                shutil.copy2(full_attachment_path, dest_asset_path)
                
                # If it's an image, use picture tag, otherwise use markdown link
                if asset_subfolder == 'img':
                    return f'{{% picture {new_attachment_path} --alt {alt_text} %}}'
                else:
                    return f'![{alt_text}]({new_attachment_path})'
            return match.group(0)

        content = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_attachment, content)
        
        # Detect and process Mermaid and PlantUML code blocks
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
        
        # Calculate relative path and create destination path
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
    parser = argparse.ArgumentParser(description="Copy Markdown files with organized asset structure.")
    parser.add_argument("source", nargs='+', help="Source file(s) or folder")
    parser.add_argument("destination", help="Destination folder")
    args = parser.parse_args()

    source = args.source[0] if len(args.source) == 1 else args.source
    copy_markdown_files(source, args.destination)

if __name__ == "__main__":
    main()