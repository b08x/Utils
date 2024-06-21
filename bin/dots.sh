#!/usr/bin/env bash

dots() {
	folders=(".config/yadm/alt/" "Workspace/syncopatedIaC/" "Utils")

	export EDITOR=$(gum choose nvim micro code gedit)

	folder=$(gum choose "dots" "${folders[@]}")

	if [[ $folder == "dots" ]]; then
	  $EDITOR $(yadm list -a |fzf --sort --preview='bat {}')
	else
	  if [[ $EDITOR == *"gedit"* ]]; then
	  	file=$(gum file -a $HOME/$folder)
	  	$EDITOR -s ${file} &>/dev/null &
	  else
	  	$EDITOR $(gum file -a $HOME/$folder)
	  fi
	fi
}

cd $HOME && dots
