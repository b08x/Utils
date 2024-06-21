#!/bin/bash

# ------------------------------------------------------------------------------
# Script Name: docker-volume-backup.sh
# Description: This script provides an interactive way to back up and restore
#              Docker volumes using the 'gum' tool for enhanced UI.
#
# Requirements:
#   - gum: Install from https://github.com/charmbracelet/gum
#   - fd:  Install from https://github.com/sharkdp/fd
#   - sd:  Install from https://github.com/chmln/sd
#   - docker:  Docker must be installed and running
#
# Usage:
#   - Backup all volumes:
#       ./docker-volume-backup.sh backup
#
#   - Backup specific volumes:
#       ./docker-volume-backup.sh backup volume1 volume2 ...
#
#   - Restore volumes (interactive selection from backups):
#       ./docker-volume-backup.sh restore
# ------------------------------------------------------------------------------

# --- Configuration ------------------------------------------------------------

# Backup directory (change if needed)
backup_dir="$HOME/LLMOS/BACKUP/docker-volumes"

# --- Functions ---------------------------------------------------------------

# Function to backup Docker volumes
# Arguments: (optional) List of volumes to back up. If none are provided,
#            all volumes on the system will be backed up.
function backup_volumes {
  local volumes_to_backup
  local backup_name

  # Get a list of all Docker volumes
  all_volumes=($(docker volume ls -q))

  # Use gum choose for interactive volume selection (if no arguments)
  if [[ $# -eq 0 ]]; then
    volumes_to_backup=$(gum choose "${all_volumes[@]}" --no-limit)
  else
    volumes_to_backup=("$@")
  fi

  # Backup each selected volume
  for volume in $volumes_to_backup; do
    backup_name="${volume}_backup_$(date +%Y-%m-%d_%H-%M-%S)"
    local backup_path="$backup_dir/$backup_name"

    echo "Backing up volume: $volume to $backup_path"
    mkdir -p "$backup_path"

    # Check if network is available (to pull alpine if needed)
    if ! ping -c 1 -W 1 google.com &>/dev/null; then
      echo "Error: Network connection unavailable. Cannot pull alpine image."
      exit 1  # Exit the script if network is not available
    fi

    docker run --rm -v "$volume":/volume -v "$backup_path":/backup alpine tar cvf "/backup/${volume}.tar" -C / volume
  done
  echo "Backup complete. Backups stored in: $backup_path"
}

# Function to restore Docker volumes
function restore_volumes {
  local backup_name
  local volumes_to_restore

  # Use gum input for backup name
  backup_name=$(gum input --placeholder "Enter the name of the backup to restore from")

  local backup_path="$backup_dir/$backup_name"
  if [[ ! -d "$backup_path" ]]; then
    echo "Error: Backup directory not found: $backup_path"
    exit 1
  fi

  # Get available volumes in the backup directory using fd and sd
  available_volumes=$(fd -t f -e .tar "$backup_path" | sd '.tar$' '')

  # Use gum filter for interactive volume selection from available backups
  volumes_to_restore=$(echo "$available_volumes" | gum filter --placeholder "Filter volumes to restore")

  if [[ -z "$volumes_to_restore" ]]; then
    echo "No volumes selected for restoration."
    exit 0
  fi

  # Restore each selected volume
  for vol_to_restore in $volumes_to_restore; do
    backup_file="$backup_path/${vol_to_restore}.tar"
    if [[ -f "$backup_file" ]]; then
      echo "Restoring volume: $vol_to_restore from $backup_file"
      docker run --rm -v "$vol_to_restore":/volume -v "$backup_path":/backup alpine tar xvf "/backup/${vol_to_restore}.tar" -C /volume
    else
      echo "Error: Backup file not found: $backup_file"
    fi
  done
}

# --- Main Script Execution ----------------------------------------------------

# Handle script termination using trap for SIGINT (Ctrl+C) and SIGTSTP (Ctrl+Z)
trap handle_exit SIGINT SIGTSTP

# Clear the screen
tput clear && tput cup 15 0

# Check for command line arguments
if [[ $# -eq 0 ]]; then

  choice=$(gum choose --cursor-prefix ">" --selected-prefix ">"  "Backup volumes" "Restore volumes")

  case "$choice" in
    "Backup volumes")
      backup_volumes
      ;;
    "Restore volumes")
      restore_volumes
      ;;
    *)
      echo "Invalid choice. Exiting."
      exit 1
      ;;
  esac
else
  # Arguments provided: Non-interactive mode
  action="$1"
  shift # Remove action from arguments

  case "$action" in
    backup)
      backup_volumes "$@"
      ;;
    restore)
      restore_volumes
      ;;
    *)
      echo "Invalid action: $action. Choose either 'backup' or 'restore'."
      exit 1
      ;;
  esac
fi
