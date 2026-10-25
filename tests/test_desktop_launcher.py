"""Exercise the desktop launcher without starting services or opening Chrome."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("ready_after,service_exists", [(1, True), (4, True), (4, False)])
def test_launcher_waits_for_one_managed_server(tmp_path, ready_after, service_exists):
    source = Path(__file__).resolve().parents[1] / "launchd" / "neural-map"
    script = source.read_text()
    mock = tmp_path / "tool"
    mock.write_text(f"#!{Path(sys.executable).resolve()}\n" + '''
import json, os, pathlib, sys
root = pathlib.Path(os.environ["LAUNCHER_TEST_ROOT"])
tool = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with (root / "calls").open("a") as f:
    f.write(json.dumps([tool, args]) + "\\n")
if tool == "plutil":
    print("test-secret")
elif tool == "curl":
    assert 'X-Auth-Token: test-secret' in sys.stdin.read()
    counter = root / "requests"
    count = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(count))
    sys.exit(0 if count >= int(os.environ["READY_AFTER"]) else 7)
elif tool == "launchctl" and args[0] == "print":
    sys.exit(0 if os.environ["SERVICE_EXISTS"] == "1" else 1)
elif tool == "open":
    assert int((root / "requests").read_text()) >= int(os.environ["READY_AFTER"])
''')
    mock.chmod(0o755)
    for tool, original in {
        "plutil": "/usr/bin/plutil", "curl": "/usr/bin/curl",
        "launchctl": "/bin/launchctl", "open": "/usr/bin/open",
        "osascript": "/usr/bin/osascript",
    }.items():
        target = tmp_path / tool
        target.symlink_to(mock)
        script = script.replace(original, str(target))
    launcher = tmp_path / "launcher"
    launcher.write_text(script)
    subprocess.run(["/bin/bash", str(launcher)], check=True, timeout=10, env={
        **os.environ, "LAUNCHER_TEST_ROOT": str(tmp_path),
        "READY_AFTER": str(ready_after), "SERVICE_EXISTS": str(int(service_exists)),
    })
    calls = [json.loads(line) for line in (tmp_path / "calls").read_text().splitlines()]
    starts = [args[0] for tool, args in calls if tool == "launchctl"]
    assert starts == ([] if ready_after == 1 else
                      ["print", "kickstart"] if service_exists else
                      ["print", "bootstrap", "kickstart"])
    assert sum(tool == "open" for tool, _ in calls) == 1
    assert not any(tool == "osascript" for tool, _ in calls)
    assert all("test-secret" not in " ".join(args) for tool, args in calls if tool == "curl")
    assert all("-k" not in args for tool, args in calls if tool == "launchctl")
