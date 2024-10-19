#!/usr/bin/env bash

# Function to start containers for a given directory and compose file (optional)
start_containers() {
  local directory="$1"
  local compose_file="$2" # Optional: Specify compose file if different from docker-compose.yml
  cd "$directory"

  # Use specified compose file or default to docker-compose.yml
  if [[ -n "$compose_file" ]]; then
    docker-compose -f "$compose_file" up -d
  else
    docker-compose up -d
  fi
  sleep 3
}

# Function to stop containers for a given directory and compose file (optional)
stop_containers() {
  local directory="$1"
  local compose_file="$2" # Optional: Specify compose file if different from docker-compose.yml
  cd "$directory"

  # Use specified compose file or default to docker-compose.yml
  if [[ -n "$compose_file" ]]; then
    docker-compose -f "$compose_file" down
  else
    docker-compose down
  fi
  sleep 3
}

# Check for command line arguments
if [[ $# -eq 0 ]]; then
  echo "Usage: $0 [start|stop]"
  exit 1
fi

# Get the action (start or stop) from the command line argument
action="$1"

case "$action" in
  start)
    # Start containers for all applications
    start_containers "$HOME/Workspace/flowbots" 
    start_containers "$HOME/LocalLLM/LLMOS/dify/docker" 
    # start_containers "$HOME/LLMOS/SillyTavern/docker" "docker-compose-standalone.yml" # Specify compose file for SillyTavern
    start_containers "$HOME/LocalLLM/LLMOS/big-AGI"
    docker container start ollama
    ;;
  stop)
    # Stop containers for all applications
    stop_containers "$HOME/Workspace/flowbots"
    stop_containers "$HOME/LocalLLM/LLMOS/dify/docker" 
    # stop_containers "$HOME/LLMOS/SillyTavern/docker" "docker-compose-standalone.yml" # Specify compose file for SillyTavern
    stop_containers "$HOME/LocalLLM/LLMOS/big-AGI"
    docker container stop ollama
    ;;
  *)
    echo "Invalid action: $action. Choose either 'start' or 'stop'."
    exit 1
    ;;
esac
