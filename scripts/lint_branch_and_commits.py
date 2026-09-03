"""Governance Linter for Branch Names and Commit Messages (42 DSLR).

Validates branch names and commit messages against institutional 42 standards.
When executed in GitHub Actions CI, automatically posts rejection feedback and
closes any Pull Request that violates branch naming or commit conventions.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

BRANCH_REGEX = re.compile(
    r"^(feat|fix|docs|test|refactor|chore|ci|build|perf)/"
    r"(dslr-[0-9]{2}|infra|hotfix|[a-z0-9-]+)(-[a-z0-9-]+)*$|^dependabot/.*$"
)

COMMIT_REGEX = re.compile(
    r"^([^:]* )?(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(\([a-zA-Z0-9_\/-]+\))?:\s*(\[([a-zA-Z0-9_:#-]+)\])?\s*(.+)$"
)

RESERVED_TAGS = {
    "INFRA",
    "CHORE",
    "DOCS",
    "FIX",
    "HOTFIX",
    "GLOBAL",
    "CONFIG",
    "SECURITY",
    "COMMUNITY",
    "DEPS",
    "RELEASE",
}


def validate_branch_name(branch_name: str) -> Tuple[bool, str]:
    """Validates branch name format against 42 DSLR naming conventions.

    Args:
        branch_name (str): Head branch name to validate.

    Returns:
        Tuple[bool, str]: (is_valid, error_message).
    """
    if not branch_name or branch_name == "main":
        return True, ""

    if not BRANCH_REGEX.match(branch_name):
        msg = (
            f"❌ **Invalid Branch Name:** `{branch_name}`\n\n"
            f"**Required Format:** `<type>/<task-id>-<short-description-in-kebab-case>`\n\n"
            f"**Valid Examples:**\n"
            f"- `feat/dslr-01-data-pipeline`\n"
            f"- `feat/dslr-02-stats-math`\n"
            f"- `fix/dslr-03-quantile-interpolation`\n"
            f"- `docs/dslr-13-peer-defense`\n"
            f"- `chore/infra-makefile-update`\n\n"
            f"**How to Fix Locally:**\n"
            f"```bash\n"
            f"git branch -m {branch_name} feat/<task-id>-<description>\n"
            f"git push origin -u feat/<task-id>-<description>\n"
            f"git push origin --delete {branch_name}\n"
            f"```"
        )
        return False, msg

    return True, ""


def validate_commit_message(commit_msg: str, issues_dir: Path) -> Tuple[bool, str]:
    """Validates single commit message against Conventional Commits and Task IDs.

    Args:
        commit_msg (str): First line of commit message.
        issues_dir (Path): Path to .github/issues/ directory.

    Returns:
        Tuple[bool, str]: (is_valid, error_message).
    """
    first_line = commit_msg.strip().splitlines()[0] if commit_msg.strip() else ""

    # Allow merge and rebase commits
    if (
        first_line.startswith("Merge ")
        or first_line.startswith("Revert ")
        or re.match(r"^[0-9]+\.[0-9]+\.[0-9]+", first_line)
    ):
        return True, ""

    match = COMMIT_REGEX.match(first_line)
    if not match:
        msg = (
            f"❌ **Invalid Commit Message:** `{first_line}`\n\n"
            f"**Required Format:**\n"
            f"- `<type>(<scope>): [<TASK-ID>:#<NUM>] <description>`\n"
            f"- or `<type>(<scope>): [<TASK-ID>] <description>`\n\n"
            f"**Valid Examples:**\n"
            f"- `feat(analytics): [DSLR-01:#1] load csv and handle missing values`\n"
            f"- `feat(math): [DSLR-02:#2] implement count and mean from scratch`\n"
            f"- `docs(theory): [DSLR-09:#9] derive binary cross-entropy loss function`\n"
            f"- `chore(build): [INFRA] configure pre-commit and ast norm checker`\n\n"
            f"**Allowed Reserved Tags:** `[INFRA]`, `[CHORE]`, `[DOCS]`, `[FIX]`, `[RELEASE]`"
        )
        return False, msg

    raw_task_tag = match.group(5)
    if raw_task_tag:
        task_tag = raw_task_tag.split(":")[0].upper()
        if task_tag not in RESERVED_TAGS and issues_dir.exists():
            lower_tag = task_tag.lower()
            matching = list(issues_dir.glob(f"*{lower_tag}*.md"))
            if not matching:
                msg = (
                    f"❌ **Unrecognized Task ID:** `[{raw_task_tag}]`\n"
                    f"Commit message: `{first_line}`\n\n"
                    f"No task matching `*{lower_tag}*.md` exists in `.github/issues/`.\n"
                    f"Please verify the task ID (e.g. `DSLR-01` to `DSLR-13`) or use `[INFRA]`."
                )
                return False, msg

    return True, ""


def get_pr_commits(base_ref: str, head_ref: str) -> List[str]:
    """Retrieves all commit messages in the current pull request range.

    Args:
        base_ref (str): Base branch or commit sha.
        head_ref (str): Head branch or commit sha.

    Returns:
        List[str]: List of commit message subjects.
    """
    if not base_ref:
        base_ref = "main"

    # Try origin/base_ref..HEAD first
    cmd = ["git", "log", f"origin/{base_ref}..HEAD", "--format=%s"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode == 0:
        return [line.strip() for line in res.stdout.splitlines() if line.strip()]

    # Fallback to local base_ref..HEAD
    cmd = ["git", "log", f"{base_ref}..HEAD", "--format=%s"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode == 0:
        return [line.strip() for line in res.stdout.splitlines() if line.strip()]

    return []


def close_pull_request(pr_number: str, rejection_reason: str) -> None:
    """Closes the Pull Request and posts the rejection reason via GitHub CLI.

    Args:
        pr_number (str): Pull Request number.
        rejection_reason (str): Markdown formatted feedback.
    """
    comment_body = (
        f"### 🛡️ 42 DSLR — Automated Governance Quality Gate\n\n"
        f"⛔ **Pull Request Closed Automatically:**\n\n"
        f"{rejection_reason}\n\n"
        f"---\n"
        f"💡 *This PR was automatically closed to enforce 42 institutional standards. "
        f"Please adjust your branch name or commit history, and reopen.*"
    )
    print("📢 Closing Pull Request due to governance violation...")
    subprocess.run(["gh", "pr", "comment", pr_number, "--body", comment_body], check=False)
    subprocess.run(["gh", "pr", "close", pr_number], check=False)


def main() -> None:
    """Main execution function for branch and commit linter."""
    parser = argparse.ArgumentParser(description="42 DSLR Branch & Commit Governance Linter")
    parser.add_argument("--branch", type=str, help="Branch name to validate")
    parser.add_argument("--base", type=str, default="main", help="Base branch reference")
    parser.add_argument("--pr-number", type=str, help="PR number for auto-closing")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    issues_dir = root_dir / ".github" / "issues"

    branch_name = args.branch or os.environ.get("GITHUB_HEAD_REF", "")
    violations: List[str] = []

    # 1. Validate Branch Name
    if branch_name:
        is_valid_branch, branch_err = validate_branch_name(branch_name)
        if not is_valid_branch:
            violations.append(branch_err)

    # 2. Validate Commit Messages
    commits = get_pr_commits(args.base, branch_name)
    for commit in commits:
        is_valid_commit, commit_err = validate_commit_message(commit, issues_dir)
        if not is_valid_commit:
            violations.append(commit_err)

    # 3. Handle Results
    if violations:
        full_report = "\n\n---\n\n".join(violations)
        print("\n" + "=" * 72)
        print(" ⛔ QUALITY GATE REJECTED: GOVERNANCE VIOLATIONS DETECTED")
        print("=" * 72)
        print(full_report)
        print("=" * 72 + "\n")

        if args.pr_number and os.environ.get("GITHUB_TOKEN"):
            close_pull_request(args.pr_number, full_report)

        sys.exit(1)

    print("✅ Quality Gate PASSED: Branch name and commits comply with 42 governance!")
    sys.exit(0)


if __name__ == "__main__":
    main()
