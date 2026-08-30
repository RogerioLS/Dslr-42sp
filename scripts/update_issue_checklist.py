#!/usr/bin/env python3
"""Automatically Check Off Issue Checkboxes and Close Tasks Upon Audit Success.

Parses PR title, description, and commit messages for task references like [DSLR-01:#1]
or 'Closes #1'. When make audit passes 100%, fetches the linked GitHub Issue,
converts all '- [ ]' checkboxes to '- [x]', posts an audit verification comment,
and closes the Issue via GitHub REST API.
"""

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = BASE_DIR / "artifacts" / "audit_summary.json"


def _extract_issue_number(event_data: dict) -> Optional[int]:
    """Extracts issue number from PR title, body, or commit messages."""
    search_texts: list[str] = []

    if "pull_request" in event_data:
        pr = event_data["pull_request"]
        search_texts.append(pr.get("title", ""))
        search_texts.append(pr.get("body", "") or "")

    if "commits" in event_data:
        for commit in event_data["commits"]:
            search_texts.append(commit.get("message", ""))

    patterns = [
        r"\[[a-zA-Z0-9_-]+:#([0-9]+)\]",
        r"(?:Closes|Close|Fixes|Fix|Resolves|Resolve)\s+#([0-9]+)",
    ]

    for text in search_texts:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))

    return None


def _github_api_request(
    url: str, method: str, token: str, data: Optional[dict] = None
) -> dict:
    """Executes an HTTP request against GitHub REST API and returns JSON response."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "42-DSLR-Issue-Automator",
    }
    encoded_data = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(
        url, data=encoded_data, headers=headers, method=method
    )

    with urllib.request.urlopen(req) as resp:
        content = resp.read().decode("utf-8")
        return json.loads(content) if content else {}


def _update_issue_checkboxes(
    repo: str, issue_number: int, token: str, metrics: dict
) -> None:
    """Checks off issue checkboxes, posts audit certificate, and closes the issue."""
    issue_url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"
    comments_url = f"{issue_url}/comments"

    try:
        issue_data = _github_api_request(issue_url, "GET", token)
    except Exception as exc:
        print(f"⚠️ Failed to fetch Issue #{issue_number}: {exc}")
        return

    body = issue_data.get("body") or ""
    current_state = issue_data.get("state")

    # Convert all unchecked boxes to checked
    new_body = body.replace("- [ ]", "- [x]")
    count_updated = body.count("- [ ]")

    # Post audit verification comment
    passed_tests = metrics.get("passed_tests", 0)
    total_tests = metrics.get("total_tests", 0)
    timestamp = metrics.get("timestamp", "N/A")

    comment_text = (
        f"### 🧙‍♂️ 42 DSLR — Automated Audit Verification\n\n"
        f"✅ **Task Audit Passed 100%** (`{timestamp}`)\n"
        f"- 🛡️ **42 Norm & Anti-Cheating**: 0 errors\n"
        f"- 🧪 **Unit Tests**: {passed_tests}/{total_tests} passed\n"
        f"- ⚡ **Python 3.10 Syntax**: Verified\n\n"
        f"All acceptance criteria validated. Issue checked off and closed automatically."
    )

    try:
        _github_api_request(comments_url, "POST", token, {"body": comment_text})
        print(f"✅ Posted audit certificate comment on Issue #{issue_number}.")
    except Exception as exc:
        print(f"⚠️ Failed to post comment on Issue #{issue_number}: {exc}")

    # Close issue and update body
    patch_data = {"body": new_body, "state": "closed"}
    try:
        _github_api_request(issue_url, "PATCH", token, patch_data)
        print(
            f"✅ Issue #{issue_number} successfully updated: "
            f"{count_updated} checkboxes checked off & state set to closed."
        )
    except Exception as exc:
        print(f"⚠️ Failed to close Issue #{issue_number}: {exc}")


def main() -> None:
    """Main entrypoint for issue checklist automation script."""
    event_path = os.getenv("GITHUB_EVENT_PATH")
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")

    if not event_path or not token or not repo or not os.path.exists(event_path):
        print(
            "ℹ️ Skipping Issue automation: "
            "Not running inside GitHub Action or missing required env vars."
        )
        return

    with open(event_path, "r", encoding="utf-8") as f:
        event_data = json.load(f)

    issue_number = _extract_issue_number(event_data)
    if not issue_number:
        print(
            "ℹ️ Generic PR / Commit detected. "
            "No task issue reference (e.g. [DSLR-01:#1]) found. Skipping."
        )
        return

    if not METRICS_PATH.exists():
        print(f"⚠️ Audit summary metrics missing at {METRICS_PATH}")
        return

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    if not metrics.get("overall_passed", False):
        print(
            f"ℹ️ Audit failed for Issue #{issue_number}. "
            "Leaving issue open and checkboxes unchecked."
        )
        return

    print(f"🚀 Processing automated checklist and closure for Issue #{issue_number}...")
    _update_issue_checkboxes(repo, issue_number, token, metrics)


if __name__ == "__main__":
    main()
