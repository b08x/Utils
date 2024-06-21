#!/bin/bash

# ------------------------------------------------------------------------------
# Script Name: remote_docker_compose_manager.sh
# Description: This script provides an interactive way to manage Docker Compose
#              services (Up/Down/Stop) on a remote server using 'gum' for enhanced UI.
#
# Requirements:
#   - gum: Install from https://github.com/charmbracelet/gum
#   - ssh:  Must be configured for passwordless access to the remote server
#   - docker-compose: Must be installed on the remote server
#
# Usage:
#   - Run the script:
#       ./remote_docker_compose_manager.sh
#
#   - The script will prompt you to choose between:
#       - Up docker-compose services
#       - Down docker-compose services
#       - Stop docker-compose services
#
#   - The script will then manage the services in each of the specified compose
#     files.
# ------------------------------------------------------------------------------

# --- Configuration ------------------------------------------------------------

# Remote host details
remote_host="b08x@ninjabot.syncopated.net"
LLMOS="$HOME/LLMOS"

# Array of compose files to manage
compose_files=(
  "$LLMOS/flowise/docker/docker-compose.yml"
  "$LLMOS/dify/docker/docker-compose.yaml"
  "$LLMOS/SillyTavern/docker/docker-compose-with-extras.yml"
)

# --- Functions ---------------------------------------------------------------

# Function to execute a command remotely via SSH
# Arguments:
#   - command: The command to execute on the remote server
#   - compose_file:  The path to the Docker Compose file
function run_remote_command {
  local command="$1"
  local compose_file="$2"
  ssh "$remote_host" "cd \"$(dirname '$compose_file')\" && docker compose -f \"$compose_file\" $command"
}

# --- Script Logic ------------------------------------------------------------
# Handle script termination using trap for SIGINT (Ctrl+C) and SIGTSTP (Ctrl+Z)
trap handle_exit SIGINT SIGTSTP

# Clear the screen
tput clear && tput cup 15 0

# Use gum choose for action selection
action=$(gum choose --cursor-prefix ">" --selected-prefix ">"  "Up docker-compose services" "Down docker-compose services" "Restart docker-compose services" "Stop docker-compose services")

# Loop through each compose file
for compose_file in "${compose_files[@]}"; do
  echo "Managing docker-compose services in file: $compose_file"

  # Determine the command based on the user's choice
  case "$action" in
    "Up docker-compose services")
      command="up -d"
      echo "Bringing docker-compose services up..."
      ;;
    "Down docker-compose services")
      command="down"
      echo "Bringing docker-compose services down..."
      ;;
    "Restart docker-compose services")
      command="restart"
      echo "Restarting docker-compose services..."
      ;;
    "Stop docker-compose services")
      command="stop"
      echo "Stopping docker-compose services..."
      ;;
    *)
      echo "Invalid action. Exiting."
      exit 1
      ;;
  esac

  # Execute the command remotely
  run_remote_command "$command" "$compose_file"

  # Check for errors and provide more specific information
  if [[ $? -ne 0 ]]; then
    echo "Error managing docker-compose services in file: $compose_file"
    echo "Check the remote server logs for more details."
    # Optionally, capture the error output from the remote server
    # error_output=$(ssh "$remote_host" "cd $(dirname '$compose_file') && docker compose -f '$compose_file' $command 2>&1")
    # echo "Error output: $error_output"
  fi

    # Add a 20-second sleep with gum spin animation
    gum spin --title "Waiting 20 Seconds Before Processing next compose file..." -- sleep 20

done

echo "Docker Compose service management completed."
