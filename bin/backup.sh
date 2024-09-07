#!/usr/bin/env bash

export timestampe=$(date +'%Y-%m-%d_%H-%M-%S')
echo $timestampe

declare -rx DELETED="/mnt/usb/backup/DELETED"

gum style \
	--foreground 014 --border-foreground 024 --border double \
	--align center --width 50 --margin "1 2" --padding "2 4" \
	'Hello.' && sleep 1 && clear

if [[ -d $DELETED ]]; then

	mkdir -pv $DELETED/$timestampe

	rsync -rtPpv --log-file=/tmp/rsync-$timestampe.log --stats --delete-before --delete-excluded --progress --ignore-existing -u -l -b -i -s \
		--suffix="_backup" --backup-dir=$DELETED/$timestampe \
		--exclude-from=$HOME/.backup_exclude.txt /home/b08x/ /mnt/bender/backup/home/b08x/

	chown -R b08x:b08x $DELETED/$timestampe
else
	echo "no usb drive mounted. exiting"
fi

# rsync -r -n -t -v --progress --ignore-existing -u -l -i -s --exclude=*.git --exclude=*.ray-snapshots/ --exclude=Recordings/*** /srv/storage/ /mnt/bender/backup/Library/
