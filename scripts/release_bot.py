"""Automated Milestone Release & Changelog Bot for 42 DSLR.

Triggered on milestone closing or manual dispatch:
1. Validates that all issues in the milestone are closed (zero open issues).
2. Extracts detailed technical tasks from each issue body for rich release notes.
3. Categorizes closed tasks into semantic groups matching .github/release.yml.
4. Automatically updates CHANGELOG.md following Keep a Changelog standard.
5. Handles protected branch fallbacks via automated PR and publishes GitHub Release.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
from typing import Any, Dict, List

MILESTONE_VERSION_MAP: Dict[str, str] = {
    "01": "v0.1.0",
    "02": "v0.2.0",
    "03": "v0.3.0",
    "04": "v1.0.0",
}

VERSION_TO_PREFIX_MAP: Dict[str, str] = {
    "0.1.0": "01",
    "0.1": "01",
    "0.2.0": "02",
    "0.2": "02",
    "0.3.0": "03",
    "0.3": "03",
    "1.0.0": "04",
    "1.0": "04",
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


def resolve_milestone_number(repo: str, raw_input: str) -> int:
    """Resolves arbitrary user input (e.g. '1', 'v0.1.0', '0.1.0') to milestone integer ID.

    Args:
        repo (str): Repository in 'owner/repo' format.
        raw_input (str): User supplied string from CLI or workflow input.

    Returns:
        int: The resolved milestone number.

    Raises:
        ValueError: If the milestone cannot be found.
    """
    clean_val = raw_input.strip()

    # Case 1: Pure integer (e.g. "1", "01")
    if clean_val.isdigit():
        return int(clean_val)

    # Fetch all milestones from repo
    milestones = run_gh_api(f"repos/{repo}/milestones?state=all&per_page=100")
    if not isinstance(milestones, list):
        milestones = []

    # Case 2: Version string (e.g. "v0.1.0", "0.1.0")
    norm_version = clean_val.lstrip("v")
    prefix = VERSION_TO_PREFIX_MAP.get(norm_version)

    for ms in milestones:
        title = ms.get("title", "")
        # Check by prefix match (e.g. "01")
        if prefix and title.startswith(prefix):
            return int(ms["number"])
        # Check by exact title match
        if clean_val.lower() in title.lower():
            return int(ms["number"])

    # Fallback to first milestone if matched
    if milestones:
        return int(milestones[0]["number"])

    raise ValueError(f"Could not resolve milestone number for input: '{raw_input}'")


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
    # Query live open issues from GitHub API instead of relying on cached counters
    open_items = run_gh_api(
        f"repos/{repo}/issues?milestone={milestone_number}&state=open&per_page=100"
    )
    if open_items:
        open_count = len(open_items)
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


def extract_task_details(body: str) -> List[str]:
    """Extracts completed technical tasks or summary bullet points from issue body.

    Args:
        body (str): Raw markdown body of the issue.

    Returns:
        List[str]: List of bullet point highlights.
    """
    if not body:
        return []

    lines = body.splitlines()
    details: List[str] = []
    in_tasks_section = False

    for line in lines:
        stripped = line.strip()
        if "Tarefas Técnicas" in stripped or "Tasks" in stripped:
            in_tasks_section = True
            continue
        if in_tasks_section and stripped.startswith("##"):
            in_tasks_section = False

        if in_tasks_section:
            clean_item = re.sub(r"^-\s*(\[[ xX]\])?\s*", "", stripped)
            if clean_item and len(clean_item) > 3:
                details.append(clean_item)
        elif stripped.startswith("- [x]") or stripped.startswith("- [X]"):
            clean_item = re.sub(r"^-\s*\[[xX]\]\s*", "", stripped)
            if clean_item and len(clean_item) > 3:
                details.append(clean_item)

    return details[:4]


def categorize_items(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Sorts milestone issues and PRs into semantic release categories.

    Args:
        items (List[Dict[str, Any]]): List of raw issue/PR dictionaries.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Categorized mapping with title, url, number, details.
    """
    categorized: Dict[str, List[Dict[str, Any]]] = {
        cat["title"]: [] for cat in CATEGORY_DEFINITIONS
    }

    for item in items:
        title = item.get("title", "").strip()
        number = item.get("number")
        html_url = item.get("html_url", "")
        author = item.get("user", {}).get("login", "")
        body = item.get("body", "") or ""
        labels = {lbl.get("name", "") for lbl in item.get("labels", [])}
        details = extract_task_details(body)

        item_obj = {
            "title": title,
            "number": str(number),
            "url": html_url,
            "author": author,
            "details": details,
        }

        placed = False
        for cat in CATEGORY_DEFINITIONS:
            if labels.intersection(cat["labels"]):
                categorized[cat["title"]].append(item_obj)
                placed = True
                break

        if not placed:
            categorized["✨ Features & Algoritmos"].append(item_obj)

    return categorized


def build_release_markdown(
    version: str, milestone_title: str, categorized: Dict[str, List[Dict[str, Any]]]
) -> str:
    """Generates rich, detailed release notes in Markdown with task bullets.

    Args:
        version (str): The semantic version (e.g. 'v0.1.0').
        milestone_title (str): Milestone title string.
        categorized (Dict[str, List[Dict[str, Any]]]): The categorized items with details.

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
            lines.append(f"- **{it['title']}** ([#{it['number']}]({it['url']})){author_credit}")
            for detail in it.get("details", []):
                lines.append(f"  - {detail}")
        lines.append("")

    return "\n".join(lines).strip()


def update_changelog_file(changelog_path: str, release_notes: str, version: str) -> None:
    """Inserts or replaces generated release notes into the CHANGELOG.md file.

    Args:
        changelog_path (str): Absolute or relative path to CHANGELOG.md.
        release_notes (str): The Markdown release section to inject.
        version (str): The semantic version (e.g. 'v0.1.0').
    """
    if not os.path.exists(changelog_path):
        raise FileNotFoundError(f"CHANGELOG.md not found at {changelog_path}")

    with open(changelog_path, "r", encoding="utf-8") as file:
        content = file.read()

    clean_ver = version.lstrip("v")
    version_pattern = rf"## \[{re.escape(clean_ver)}\][\s\S]*?(?=\n## \[|\Z)"
    content_cleaned = re.sub(version_pattern, "", content)
    content_cleaned = re.sub(r"(---\s*){2,}", "---\n\n", content_cleaned)

    unreleased_marker = "## [Unreleased]"
    if unreleased_marker in content_cleaned:
        parts = content_cleaned.split(unreleased_marker, 1)
        remaining = parts[1].lstrip().lstrip("-").lstrip()
        new_content = (
            f"{parts[0]}{unreleased_marker}\n\n---\n\n" f"{release_notes}\n\n---\n\n" f"{remaining}"
        )
    else:
        new_content = f"{content_cleaned}\n\n---\n\n{release_notes}\n"

    with open(changelog_path, "w", encoding="utf-8") as file:
        file.write(new_content)


def commit_and_create_release(
    version: str, milestone_title: str, release_notes: str, dry_run: bool = False
) -> None:
    """Commits updated CHANGELOG.md and creates GitHub release with protected branch handling.

    Args:
        version (str): The version tag (e.g. 'v0.1.0').
        milestone_title (str): The milestone title.
        release_notes (str): Markdown text for the release.
        dry_run (bool): If True, does not commit or publish to GitHub.
    """
    if dry_run:
        print("🔍 [DRY RUN] Rich release notes generated successfully:\n")
        print(release_notes)
        return

    # Git configuration for CI bot
    subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=False)
    subprocess.run(
        ["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"],
        check=False,
    )

    # Git Add & Commit CHANGELOG.md
    subprocess.run(["git", "add", "CHANGELOG.md"], check=True)
    commit_msg = f"chore(release): [RELEASE] publish release {version} for {milestone_title}"
    subprocess.run(["git", "commit", "-m", commit_msg], check=False)

    token = os.environ.get("GITHUB_TOKEN")
    env = os.environ.copy()
    if token:
        env["GH_TOKEN"] = token

    # Attempt direct push to main
    push_res = subprocess.run(
        ["git", "push", "origin", "main"], capture_output=True, text=True, check=False
    )
    if push_res.returncode != 0:
        print(
            "ℹ️ Direct push to main blocked by branch protection. "
            "Opening automated PR for CHANGELOG.md..."
        )
        branch_name = f"chore/release-{version.replace('.', '-')}-changelog"
        subprocess.run(["git", "checkout", "-b", branch_name], check=False)
        subprocess.run(["git", "push", "origin", branch_name, "--force"], check=False)

        pr_cmd = [
            "gh",
            "pr",
            "create",
            "--title",
            commit_msg,
            "--body",
            f"Automated PR to update `CHANGELOG.md` with release notes for **{version}**.",
            "--base",
            "main",
            "--head",
            branch_name,
        ]
        subprocess.run(pr_cmd, env=env, check=False)
        # Attempt merge
        subprocess.run(
            ["gh", "pr", "merge", branch_name, "--auto", "--merge"], env=env, check=False
        )

    # Create or update GitHub Release
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
    result = subprocess.run(release_cmd, capture_output=True, text=True, env=env, check=False)
    if result.returncode != 0:
        # If release already exists, edit it with new rich notes
        edit_cmd = ["gh", "release", "edit", version, "--notes", release_notes]
        subprocess.run(edit_cmd, capture_output=True, text=True, env=env, check=False)

    print(f"🎉 Release {version} e CHANGELOG.md processados com sucesso no GitHub!")


def main() -> None:
    """Main CLI entry point for release_bot."""
    parser = argparse.ArgumentParser(description="42 DSLR Automated Milestone Release Bot")
    parser.add_argument(
        "--milestone-number",
        "--milestone",
        dest="milestone_target",
        type=str,
        required=True,
        help="The GitHub milestone number (e.g. '1') or version tag (e.g. 'v0.1.0')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate generation without writing or publishing",
    )
    args = parser.parse_args()

    repo = get_current_repo()
    milestone_num = resolve_milestone_number(repo, args.milestone_target)

    print(
        f"🤖 [RELEASE BOT] Resolvido '{args.milestone_target}' -> "
        f"Milestone #{milestone_num} no repo '{repo}'..."
    )

    data = fetch_milestone_and_issues(repo, milestone_num)
    milestone_title = data["milestone"].get("title", f"Milestone #{milestone_num}")
    version = determine_version(milestone_title)

    categorized = categorize_items(data["items"])
    notes = build_release_markdown(version, milestone_title, categorized)

    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    changelog_path = os.path.join(workspace_root, "CHANGELOG.md")

    if not args.dry_run:
        update_changelog_file(changelog_path, notes, version)

    commit_and_create_release(version, milestone_title, notes, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
