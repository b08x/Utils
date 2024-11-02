#!/bin/bash

set -e

maim -u -s | xclip -selection clipboard -t image/png
