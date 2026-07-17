#!/bin/sh
# Daily portable export of the Neural Map, plus a copy off the external volume.
#
# Writes memory_store/backups/daily/neural-map-YYYY-MM-DD.json and mirrors it to
# /Users/Shared/neuralmap-backups on the boot volume, so a dead external drive
# doesn't take the memories with it. Keeps 14 days in each location.
#
# Scheduled from cron (see `crontab -l`); safe to run by hand any time.
set -eu
umask 077

PROJECT="$(cd "$(dirname "$0")" && pwd)"
DAILY="$PROJECT/memory_store/backups/daily"
OFFVOLUME="/Users/Shared/neuralmap-backups"
STAMP="$(date +%Y-%m-%d)"
OUT="$DAILY/neural-map-$STAMP.json"

mkdir -p "$DAILY" "$OFFVOLUME"
chmod 700 "$DAILY" "$OFFVOLUME"

"$PROJECT/.venv/bin/python" "$PROJECT/backup.py" export -o "$OUT"
cp "$OUT" "$OFFVOLUME/"
chmod 600 "$OUT" "$OFFVOLUME/neural-map-$STAMP.json"

find "$DAILY" -name 'neural-map-*.json' -mtime +14 -delete
find "$OFFVOLUME" -name 'neural-map-*.json' -mtime +14 -delete

echo "$(date '+%Y-%m-%d %H:%M:%S') backed up to $OUT and $OFFVOLUME/"
