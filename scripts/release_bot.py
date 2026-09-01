"""Automated Milestone Release & Changelog Bot for 42 DSLR.

Triggered on milestone closing or manual dispatch:
1. Validates that all issues in the milestone are closed (zero open issues).
2. Categorizes closed tasks into semantic groups matching .github/release.yml.
3. Automatically updates CHANGELOG.md following Keep a Changelog standard.
4. Commits and creates the official GitHub Release with semantic tag.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List

MILESTONE_VERSION_MAP: Dict[str, str] = {
    "01": "v0.1.0",
    "02": "v0.2.0",
    "03": "v0.3.0",
    "04": "v1.0.0",
}

CATEGORY_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "title": "✨ Features & Algoritmos",
        "labels": {"type: implementation", "area: model", "area: stats", "area: preprocessing"},
    },
    {
        "title": "📈 Visualizações Gráficas",
        "labels": {"area: visualization"},
    },
    {
        "title": "🛡️ Normas, Testes & Qualidade",
        "labels": {"type: test", "type: defense", "area: devops"},
    },
    {
        "title": "📚 Documentação & Matemática",
        "labels": {"type: docs", "type: math-heavy", "type: pedagogical"},
    },
    {
        "title": "🐛 Correções de Bugs",
        "labels": {"type: bug", "fix"},
    },
]


def run_gh_api(endpoint: str) -> Any:
    """Executes a GitHub API call using the official gh CLI tool.

    Args:
        endpoint (str): The relative GitHub API endpoint (e.g. 'repos/{owner}/{repo}/...').

    Returns:
        Any: Decoded JSON response from GitHub API.

    Raises:
        RuntimeError: If the gh command fails or output is invalid JSON.
    """
    token = os.environ.get("GITHUB_TOKEN")
    cmd = ["gh", "api", endpoint]
    env = os.environ.copy()
    if token:
        env["GH_TOKEN"] = token

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"GitHub API call failed for '{endpoint}': {result.stderr.strip()}")

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as err:
        raise RuntimeError(f"Failed to parse JSON response from '{endpoint}': {err}") from err


def get_current_repo() -> str:
    """Detects current repository in 'owner/repo' format.

    Returns:
        str: Repository name with owner.
    """
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        return repo

    result = subprocess.run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return "RogerioLS/Dslr-42sp"


def determine_version(milestone_title: str) -> str:
    """Maps milestone title prefix to a semantic version tag.

    Args:
        milestone_title (str): The milestone title string.

    Returns:
        str: Tag formatted version string (e.g. 'v0.1.0').
    """
    match = re.match(r"^(\d{2})", milestone_title.strip())
    if match:
        prefix = match.group(1)
        if prefix in MILESTONE_VERSION_MAP:
            return MILESTONE_VERSION_MAP[prefix]

    semver_match = re.search(r"v?(\d+\.\d+\.\d+)", milestone_title)
    if semver_match:
        return f"v{semver_match.group(1)}"

    return "v0.1.0"


def fetch_milestone_and_issues(repo: str, milestone_number: int) -> Dict[str, Any]:
    """Fetches milestone metadata and all associated closed issues and PRs.

    Args:
        repo (str): Repository in 'owner/repo' format.
        milestone_number (int): The milestone number.

    Returns:
        Dict[str, Any]: Consolidated dictionary with milestone info and items.
    """
    milestone_data = run_gh_api(f"repos/{repo}/milestones/{milestone_number}")
    open_count = milestone_data.get("open_issues", 0)
    if open_count > 0:
        raise ValueError(
            f"❌ Bloqueio de Release: O Milestone '{milestone_data.get('title')}' "
            f"ainda possui {open_count} task(s) em aberto! "
            f"Todas as tasks devem ser concluídas e fechadas antes de publicar a release."
        )

    # Fetch closed issues & PRs
    items = run_gh_api(
        f"repos/{repo}/issues?milestone={milestone_number}&state=closed&per_page=100"
    )
    return {"milestone": milestone_data, "items": items}


def categorize_items(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    """Sorts milestone issues and PRs into semantic release categories.

    Args:
        items (List[Dict[str, Any]]): List of raw issue/PR dictionaries.

    Returns:
        Dict[str, List[Dict[str, str]]]: Categorized mapping with title, url, number.
    """
    categorized: Dict[str, List[Dict[str, str]]] = {
        cat["title"]: [] for cat in CATEGORY_DEFINITIONS
    }

    for item in items:
        title = item.get("title", "").strip()
        number = item.get("number")
        html_url = item.get("html_url", "")
        author = item.get("user", {}).get("login", "")
        labels = {lbl.get("name", "") for lbl in item.get("labels", [])}

        placed = False
        for cat in CATEGORY_DEFINITIONS:
            if labels.intersection(cat["labels"]):
                categorized[cat["title"]].append(
                    {
                        "title": title,
                        "number": str(number),
                        "url": html_url,
                        "author": author,
                    }
                )
                placed = True
                break

        if not placed:
            categorized["✨ Features & Algoritmos"].append(
                {
                    "title": title,
                    "number": str(number),
                    "url": html_url,
                    "author": author,
                }
            )

    return categorized


def build_release_markdown(
    version: str, milestone_title: str, categorized: Dict[str, List[Dict[str, str]]]
) -> str:
    """Generates the formatted release notes in Markdown.

    Args:
        version (str): The semantic version (e.g. 'v0.1.0').
        milestone_title (str): Milestone title string.
        categorized (Dict[str, List[Dict[str, str]]]): The categorized items.

    Returns:
        str: Markdown formatted release notes.
    """
    date_str = datetime.date.today().isoformat()
    lines: List[str] = [
        f"## [{version.lstrip('v')}] - {date_str} — {milestone_title}\n",
    ]

    for cat_title, items in categorized.items():
        if not items:
            continue
        lines.append(f"### {cat_title}")
        for it in items:
            author_credit = f" by @{it['author']}" if it["author"] else ""
            lines.append(f"- {it['title']} ([#{it['number']}]({it['url']})){author_credit}")
        lines.append("")

    return "\n".join(lines).strip()


def update_changelog_file(changelog_path: str, release_notes: str) -> None:
    """Inserts newly generated release notes into the CHANGELOG.md file.

    Args:
        changelog_path (str): Absolute or relative path to CHANGELOG.md.
        release_notes (str): The Markdown release section to inject.
    """
    if not os.path.exists(changelog_path):
        raise FileNotFoundError(f"CHANGELOG.md not found at {changelog_path}")

    with open(changelog_path, "r", encoding="utf-8") as file:
        content = file.read()

    unreleased_marker = "## [Unreleased]"
    if unreleased_marker in content:
        parts = content.split(unreleased_marker, 1)
        new_content = (
            f"{parts[0]}{unreleased_marker}\n\n---\n\n" f"{release_notes}\n" f"{parts[1].lstrip()}"
        )
    else:
        new_content = f"{content}\n\n---\n\n{release_notes}\n"

    with open(changelog_path, "w", encoding="utf-8") as file:
        file.write(new_content)


def commit_and_create_release(
    version: str, milestone_title: str, release_notes: str, dry_run: bool = False
) -> None:
    """Commits updated CHANGELOG.md and triggers GitHub release creation.

    Args:
        version (str): The version tag (e.g. 'v0.1.0').
        milestone_title (str): The milestone title.
        release_notes (str): Markdown text for the release.
        dry_run (bool): If True, does not commit or publish to GitHub.
    """
    if dry_run:
        print("🔍 [DRY RUN] Release notes generated successfully:\n")
        print(release_notes)
        return

    # Git configuration for CI bot
    subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=False)
    subprocess.run(
        ["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"],
        check=False,
    )

    # Git Add & Commit
    subprocess.run(["git", "add", "CHANGELOG.md"], check=True)
    commit_msg = f"chore(release): [RELEASE] publish release {version} for {milestone_title}"
    subprocess.run(["git", "commit", "-m", commit_msg], check=False)
    subprocess.run(["git", "push", "origin", "main"], check=False)

    # Create GitHub Release
    release_cmd = [
        "gh",
        "release",
        "create",
        version,
        "--title",
        f"{version} — {milestone_title}",
        "--notes",
        release_notes,
    ]
    token = os.environ.get("GITHUB_TOKEN")
    env = os.environ.copy()
    if token:
        env["GH_TOKEN"] = token

    result = subprocess.run(release_cmd, capture_output=True, text=True, env=env, check=False)
    if result.returncode == 0:
        print(f"🎉 Release {version} publicada com sucesso no GitHub!")
    else:
        print(f"⚠️ Aviso ao criar release via gh: {result.stderr.strip()}", file=sys.stderr)


def main() -> None:
    """Main CLI entry point for release_bot."""
    parser = argparse.ArgumentParser(description="42 DSLR Automated Milestone Release Bot")
    parser.add_argument(
        "--milestone-number",
        type=int,
        required=True,
        help="The GitHub milestone number (e.g. 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate generation without writing or publishing",
    )
    args = parser.parse_args()

    repo = get_current_repo()
    print(f"🤖 [RELEASE BOT] Processando Milestone #{args.milestone_number} no repo '{repo}'...")

    data = fetch_milestone_and_issues(repo, args.milestone_number)
    milestone_title = data["milestone"].get("title", f"Milestone #{args.milestone_number}")
    version = determine_version(milestone_title)

    categorized = categorize_items(data["items"])
    notes = build_release_markdown(version, milestone_title, categorized)

    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    changelog_path = os.path.join(workspace_root, "CHANGELOG.md")

    if not args.dry_run:
        update_changelog_file(changelog_path, notes)

    commit_and_create_release(version, milestone_title, notes, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
