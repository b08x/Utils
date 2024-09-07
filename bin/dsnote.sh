#!/bin/bash

set -e


handle_exit() {
  dsnote --action stop-listening
}


dsnote --action start-listening-active-window


# Handle script termination using trap for SIGINT (Ctrl+C) and SIGTSTP (Ctrl+Z)
trap handle_exit SIGINT SIGTSTP
