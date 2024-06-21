#!/usr/bin/env
#
#
termdown -b -c 5 10s --exec-cmd "if [ '{0}' > '5' ];
then notify-send -t 1000 -u critical '{1}'; fi"
