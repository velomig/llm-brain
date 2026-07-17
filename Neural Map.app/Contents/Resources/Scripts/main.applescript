-- Neural Map launcher: ensure the background server is up, then open the UI.
-- No Terminal window appears; everything runs via `do shell script`.
use scripting additions

set theURL to "http://127.0.0.1:8000"
set tokenURL to theURL
set buildURL to theURL & "/api/build"
set buildAuth to ""
set accountNumber to (do shell script "id -u")
set agentLabel to "com.neuralmap.server"
set agentPlist to "/Library/LaunchAgents/com.neuralmap.server.plist"
set appBundlePath to POSIX path of (path to me)
if appBundlePath ends with "/" then set appBundlePath to text 1 thru -2 of appBundlePath
set projectDir to (do shell script "/usr/bin/dirname " & quoted form of appBundlePath)
set pythonPath to projectDir & "/.venv/bin/python"

-- If the server is protected with NEURAL_MAP_TOKEN, open a one-time bootstrap
-- URL so the browser receives the nm_token cookie. The server immediately
-- redirects back to the clean URL without the token in the address bar.
try
	set serverToken to (do shell script "plutil -extract EnvironmentVariables.NEURAL_MAP_TOKEN raw -o - " & quoted form of agentPlist & " 2>/dev/null || true")
	if serverToken is not "" then
		set tokenURL to theURL & "/?token=" & serverToken
		set buildAuth to " -H " & quoted form of ("X-Auth-Token: " & serverToken)
	end if
end try

-- Is the server already answering? `curl -f` treats 401/500 as down; the old
-- launcher used plain curl, so an unauthorized response looked like success.
set serverUp to true
try
	do shell script "curl -fs -o /dev/null --max-time 2 " & quoted form of tokenURL
on error
	set serverUp to false
end try

-- An answering process can still be stale: static app.js is read from disk on
-- every reload, while FastAPI routes remain frozen in the old Python process.
-- Compare a content fingerprint and force-restart launchd only when needed.
if serverUp then
	set serverStale to false
	try
		set fingerprintCode to "from brain.build_info import source_fingerprint; print(source_fingerprint())"
		set parseCode to "import json,sys; print(json.load(sys.stdin).get('source_fingerprint',''))"
		set diskFingerprint to (do shell script "cd " & quoted form of projectDir & " && " & quoted form of pythonPath & " -c " & quoted form of fingerprintCode)
		set runningFingerprint to (do shell script "curl -fs --max-time 2" & buildAuth & " " & quoted form of buildURL & " | " & quoted form of pythonPath & " -c " & quoted form of parseCode)
		if runningFingerprint is not diskFingerprint then set serverStale to true
	on error
		-- Old servers have no /api/build route, so a 404 correctly lands here.
		set serverStale to true
	end try
	if serverStale then
		try
			do shell script "launchctl kickstart -k gui/" & accountNumber & "/" & agentLabel
		end try
		set serverUp to false
	end if
end if

-- If the service was unloaded, kickstart alone cannot recover it. Bootstrap the
-- installed plist when the label is absent, then kick it and wait for startup.
if not serverUp then
	try
		set serviceTarget to "gui/" & accountNumber & "/" & agentLabel
		do shell script "launchctl print " & serviceTarget & " >/dev/null 2>&1 || launchctl bootstrap gui/" & accountNumber & " " & quoted form of agentPlist & "; launchctl kickstart " & serviceTarget
	end try
	repeat 20 times
		try
			do shell script "curl -fs -o /dev/null --max-time 1 " & quoted form of tokenURL
			set serverUp to true
			exit repeat
		on error
			delay 1
		end try
	end repeat
end if

if serverUp then
	try
		-- Pass the URL to Chrome in one operation. Activating Chrome first and
		-- then creating/retargeting a window races Chrome's own startup window,
		-- which can leave one blank window beside the Neural Map window.
		do shell script "/usr/bin/open -a " & quoted form of "Google Chrome" & " " & quoted form of tokenURL
	on error
		open location tokenURL
	end try
else
	display dialog "Neural Map server did not respond on port 8000." & return & "Check the logs at ~/Library/Logs/neuralmap/." buttons {"OK"} default button "OK" with icon stop with title "Neural Map"
end if
