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
  sleep 5
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
  sleep 5
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
    start_containers "$HOME/LocalLLM/LLMOS/big-AGI"
    start_containers "$HOME/LocalLLM/SillyTavern/docker"
    start_containers "$HOME/LocalLLM/SillyTavern-Extras/docker"
    start_containers "$HOME/LocalLLM/lobe-chat/docker-compose/local"
    docker container start ollama
    docker container start local-ai
    ;;
  stop)
    # Stop containers for all applications
    stop_containers "$HOME/Workspace/flowbots" 
    stop_containers "$HOME/LocalLLM/LLMOS/dify/docker" 
    stop_containers "$HOME/LocalLLM/LLMOS/big-AGI"
    stop_containers "$HOME/LocalLLM/SillyTavern/docker"
    stop_containers "$HOME/LocalLLM/SillyTavern-Extras/docker"
    stop_containers "$HOME/LocalLLM/lobe-chat/docker-compose/local"
    docker container stop ollama
    docker container stop local-ai
    ;;
  *)
    echo "Invalid action: $action. Choose either 'start' or 'stop'."
    exit 1
    ;;
esac
