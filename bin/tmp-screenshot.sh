#!/bin/bash

set -e

mkdir -p "/tmp/screenshots"

FILE="/tmp/screenshots/$(date +%s).png"

screenshot-and-open "$FILE"
