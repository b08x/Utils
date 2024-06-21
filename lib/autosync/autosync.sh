/mnt/autosync#!/bin/sh
# Script to do a two-way sync with a USB drive

if [ $(mount | grep -c /mnt/autosync) != 1 ]
then
  /sbin/mount /dev/disk/by-uuid/5732DF544868E675 || exit 1
  echo "/mnt/autosync is now mounted"

  rsync -rtuv --modify-window=1 --size-only --exclude ".csync*" --exclude ".owncloud*" /mnt/autosync/ /srv/share/media/
  rsync -rtuv --modify-window=1 --size-only --exclude ".csync*" --exclude ".owncloud*" --delete /srv/share/media/ /mnt/autosync/

  sync

  # sudo -u bob DISPLAY=:0 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus notify-send "SYNCSHARE Finished Synchronizing!"

  sleep 5

  umount /mnt/autosync
else
  rsync -rtuv --modify-window=1 --size-only --exclude ".csync*" --exclude ".owncloud*" /mnt/autosync/ /srv/share/media/
  rsync -rtuv --modify-window=1 --size-only --exclude ".csync*" --exclude ".owncloud*" --delete-delay /srv/share/media/ /mnt/autosync/

  sync

  # sudo -u bob DISPLAY=:0 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus notify-send "SYNCSHARE Finished Synchronizing!"

  sleep 5

  umount /mnt/autosync
fi
