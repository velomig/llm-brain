"""Connect Neural Map to Google Calendar using an OAuth Desktop app client."""
from __future__ import annotations

import argparse
import json

from brain.toolkit.google_calendar import GoogleCalendarUnavailable, authorize_interactive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true", help="print the authorization URL without opening another browser")
    args = parser.parse_args()
    try:
        result = authorize_interactive(open_browser=not args.no_browser)
    except GoogleCalendarUnavailable as exc:
        print(f"Could not connect Google Calendar: {exc}")
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
