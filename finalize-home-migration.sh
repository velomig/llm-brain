#!/bin/zsh

set -euo pipefail

source_home="/Volumes/ExtHD4T/Users/miguelfolch"
target_home="/Users/miguelfolch"
expected_old_home="/Volumes/ExtHD4T/Users/miguelfolch"
account_name="miguelfolch"
exclude_file="${0:A:h}/migration-excludes.txt"

if [[ "$(/usr/bin/id -u)" -ne 0 ]]; then
  print -u2 "Run this finalizer with sudo from the administrator account."
  exit 1
fi

account_uid="$(/usr/bin/dscl . -read "/Users/$account_name" UniqueID | /usr/bin/awk '{print $2}')"
account_gid="$(/usr/bin/dscl . -read "/Users/$account_name" PrimaryGroupID | /usr/bin/awk '{print $2}')"
if [[ "$account_uid" != "501" ]] || [[ "$account_gid" != "20" ]]; then
  print -u2 "Safety check failed: unexpected account identity UID:GID $account_uid:$account_gid."
  exit 1
fi

source_uid="$(/usr/bin/stat -f %u "$source_home")"
target_uid="$(/usr/bin/stat -f %u "$target_home")"
if [[ "$source_uid" != "$account_uid" ]] || [[ "$target_uid" != "$account_uid" ]]; then
  print "Correcting home-root ownership (source=$source_uid target=$target_uid expected=$account_uid)..."
  /usr/sbin/diskutil enableOwnership /Volumes/ExtHD4T >/dev/null 2>&1 || true
  /usr/sbin/chown "$account_uid:$account_gid" "$source_home" "$target_home"
  source_uid="$(/usr/bin/stat -f %u "$source_home")"
  target_uid="$(/usr/bin/stat -f %u "$target_home")"
fi

if [[ "$source_uid" != "$account_uid" ]] || [[ "$target_uid" != "$account_uid" ]]; then
  print -u2 "Safety check failed after correction: source=$source_uid target=$target_uid expected=$account_uid."
  exit 1
fi

if /bin/launchctl print "gui/$account_uid" >/dev/null 2>&1; then
  print -u2 "Safety check failed: miguelfolch still has an active GUI login session."
  print -u2 "Log out fully (do not use fast user switching), then try again."
  exit 1
fi

if ! /bin/ls -A "$source_home/Library/Mail" >/dev/null 2>&1; then
  print -u2 "Safety check failed: Terminal does not have Full Disk Access."
  print -u2 "Enable Terminal in System Settings > Privacy & Security > Full Disk Access,"
  print -u2 "quit and reopen Terminal, then run this finalizer again."
  exit 1
fi

if /usr/bin/pgrep -u "$account_uid" >/dev/null 2>&1; then
  print "Stopping logged-out miguelfolch background services..."
  /bin/launchctl bootout "user/$account_uid" >/dev/null 2>&1 || true
  /bin/sleep 3
fi

if /usr/bin/pgrep -u "$account_uid" >/dev/null 2>&1; then
  print -u2 "Safety check failed: UID $account_uid processes remained after service shutdown."
  /usr/bin/pgrep -l -u "$account_uid" >&2 || true
  exit 1
fi

current_home="$(/usr/bin/dscl . -read "/Users/$account_name" NFSHomeDirectory | /usr/bin/awk '{print $2}')"
if [[ "$current_home" != "$expected_old_home" && "$current_home" != "$target_home" ]]; then
  print -u2 "Safety check failed: unexpected account home: $current_home"
  exit 1
fi

if [[ ! -f "$exclude_file" ]]; then
  print -u2 "Missing exclusion manifest: $exclude_file"
  exit 1
fi

print "Final synchronization of small home files..."
/usr/bin/rsync -aH --delete \
  --exclude-from="$exclude_file" \
  "$source_home/" "$target_home/"

print "Final synchronization of Library..."
/usr/bin/rsync -aH --delete \
  --exclude='Containers/com.docker.docker/Data/vms/0/data/Docker.raw' \
  --exclude='Application Support/FileProvider/' \
  --exclude='*.sock' \
  "$source_home/Library/" "$target_home/Library/"

source_docker="$source_home/Library/Containers/com.docker.docker/Data/vms/0/data/Docker.raw"
target_docker="$target_home/Library/Containers/com.docker.docker/Data/vms/0/data/Docker.raw"
source_docker_layout="$(/usr/bin/stat -f '%z:%b' "$source_docker")"
target_docker_layout="$(/usr/bin/stat -f '%z:%b' "$target_docker")"
target_docker_owner="$(/usr/bin/stat -f '%u:%g' "$target_docker")"
if [[ "$source_docker_layout" != "$target_docker_layout" ]] || \
   [[ "$target_docker_owner" != "$account_uid:$account_gid" ]]; then
  print -u2 "Safety check failed: Docker sparse disk does not match."
  print -u2 "source-layout=$source_docker_layout target-layout=$target_docker_layout"
  print -u2 "target-owner=$target_docker_owner expected=$account_uid:$account_gid"
  exit 1
fi

external_names=(
  .cache .lmstudio .ollama .unsloth
  "Calibre Library" ComfyUI Desktop Documents Downloads Fooocus
  Movies Music Pictures anythingllm-docs docker-apps hermes_studio
  miniforge3 open-webui-main openwebui-env
)

for name in "${external_names[@]}"; do
  link_path="$target_home/$name"
  expected_target="$source_home/$name"
  if [[ ! -d "$expected_target" ]]; then
    print -u2 "Safety check failed: missing external-data source: $expected_target"
    exit 1
  fi
  if [[ -e "$link_path" || -L "$link_path" ]]; then
    if [[ ! -L "$link_path" ]] || [[ "$(/usr/bin/readlink "$link_path")" != "$expected_target" ]]; then
      print -u2 "Safety check failed: occupied external-data link path: $link_path"
      exit 1
    fi
  else
    print "Restoring external-data link: $name"
    /bin/ln -s "$expected_target" "$link_path"
    /usr/sbin/chown -h "$account_uid:$account_gid" "$link_path"
  fi
done

if [[ "$current_home" == "$expected_old_home" ]]; then
  print "Switching the account home to the internal SSD..."
  /usr/bin/dscl . -change "/Users/$account_name" NFSHomeDirectory \
    "$expected_old_home" "$target_home"
fi

new_home="$(/usr/bin/dscl . -read "/Users/$account_name" NFSHomeDirectory | /usr/bin/awk '{print $2}')"
if [[ "$new_home" != "$target_home" ]]; then
  print -u2 "Account switch verification failed: $new_home"
  exit 1
fi

print
print "Migration finalized successfully."
print "The account home is now $target_home."
print "Large data remains on ExtHD4T through verified symbolic links."
print "The external source home has not been deleted and remains the rollback copy."
print "Restart the Mac, then log in as miguelfolch."
