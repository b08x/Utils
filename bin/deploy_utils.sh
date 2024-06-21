#!/bin/sh

declare ANSIBLE_HOME="$HOME/Workspace/syncopatedOS/cac"

ansible-playbook -i "${ANSIBLE_HOME}/inventory.ini" "${ANSIBLE_HOME}/playbooks/utils.yml" \
--limit tinybot,soundbot,ninjabot

# aplaybook -vv -i inventory.ini playbooks/utils.yml --limit tinybot,soundbot,ninjabot -e "branch=development"
