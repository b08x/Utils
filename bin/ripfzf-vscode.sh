#!/usr/bin/env bash
# ~/scripts/ripfzfsubl
#
# Uses Sublime Text (subl) as the text editor
#
# brew install bat ripgrep fzf
#
# bat  Clone of cat(1) with syntax highlighting and Git integration
#      |__ https://github.com/sharkdp/bat
#
# ripgrep  Search tool like grep and The Silver Searcher
#          |__ https://github.com/BurntSushi/ripgrep
#
# fzf  Command-line fuzzy finder written in Go
#      |__ https://github.com/junegunn/fzf
#
#
# 1. Search for text in files using Ripgrep
# 2. Interactively narrow down the list using fzf
# 3. Open the file in Sublime Text Editor
#
# source: https://github.com/MadBomber/scripts

rga --rga-accurate --rga-adapters='poppler,pandoc,zip,decompress,tar,sqlite,ffmpeg' -j4 --color=always --line-number --no-heading --smart-case "${*:-}" |
	fzf -e --ansi \
		--color "hl:-1:underline,hl+:-1:underline:reverse" \
		--delimiter : \
		--preview 'bat --color=always {1} --highlight-line {2}' \
		--preview-window '60%:wrap,border-bottom,+{2}+3/3,~3' \
		--bind 'enter:become(code -g {1}:{2})'
