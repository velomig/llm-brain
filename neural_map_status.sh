#!/bin/zsh

set -u

account_uid="$(/usr/bin/id -u)"
server_label="com.neuralmap.server"
watcher_label="com.neuralmap.vaultwatch"
server_plist="/Library/LaunchAgents/com.neuralmap.server.plist"
failed=0

agent_status() {
  local label="$1"
  local snapshot
  local state
  local pid

  if ! snapshot="$(/bin/launchctl print "gui/$account_uid/$label" 2>/dev/null)"; then
    print "$label: not loaded"
    failed=1
    return
  fi

  state="$(print -r -- "$snapshot" | /usr/bin/awk '/^[[:space:]]*state = / { print $3; exit }')"
  pid="$(print -r -- "$snapshot" | /usr/bin/awk '/^[[:space:]]*pid = / { print $3; exit }')"
  if [[ "$state" == "running" && "$pid" == <-> ]]; then
    print "$label: running (pid $pid)"
  else
    print "$label: unhealthy (state=${state:-unknown}, pid=${pid:-none})"
    failed=1
  fi
}

agent_status "$server_label"
agent_status "$watcher_label"

if [[ -r "$server_plist" ]]; then
  server_token="$(/usr/bin/plutil -extract EnvironmentVariables.NEURAL_MAP_TOKEN raw -o - "$server_plist" 2>/dev/null || true)"
else
  server_token=""
fi

curl_args=(--silent --output /dev/null --write-out '%{http_code}' --max-time 3)
if [[ -n "$server_token" ]]; then
  curl_args+=(--header "X-Auth-Token: $server_token")
fi
http_code="$(/usr/bin/curl "${curl_args[@]}" http://127.0.0.1:8000/api/health 2>/dev/null || true)"

if [[ "$http_code" == "200" ]]; then
  print "Neural Map API: healthy (HTTP 200)"
else
  print "Neural Map API: unavailable (HTTP ${http_code:-none})"
  failed=1
fi

exit "$failed"
