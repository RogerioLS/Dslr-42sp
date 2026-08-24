# Security Policy

## Overview

This repository contains the Machine Learning project for the **42 Specialization — DSLR (Data Science × Logistic Regression)**. We take code quality, safety, and responsible disclosure seriously.

---

## Supported Versions

Only the active Python version defined by the 42 curriculum is officially supported for security updates and exercise validation:

| Version | Supported |
| ------- | --------- |
| Python 3.10.x | :white_check_mark: Yes |
| Python < 3.10 | :x: No |

---

## Security Best Practices in this Repository

When contributing or reviewing code in this repository, ensure the following security standards are met:

1. **No Hardcoded Credentials or Secrets:**
   - Never commit API keys, personal access tokens, passwords, or personal credentials.
   - All secret detection is enforced pre-commit via `detect-secrets`.

2. **Safe Input Validation & Robust Error Handling:**
   - Validate and sanitize input data (CLI arguments, CSV file paths, column names).
   - Prevent arbitrary command execution or unsafe file path traversals.
   - Handle exceptions gracefully without leaking sensitive stack traces or environment details.

3. **Dependency Integrity:**
   - Keep external dependencies (NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn) updated to versions free of known CVEs.
   - Automatic scanning and dependency updates are handled by Dependabot.

4. **Environment Isolation:**
   - Always run the project inside an isolated virtual environment (`venv` or `conda`).
   - Avoid running analytical scripts with root/administrator privileges.

---

## Reporting a Vulnerability

If you discover a security vulnerability or accidental exposure of sensitive data within this repository, please report it responsibly:

1. **Do NOT open a public GitHub issue.**
2. Submit a private security advisory via GitHub or contact the repository owner ([@RogerioLS](https://github.com/RogerioLS)).
3. Include detailed steps to reproduce the issue, along with any relevant code snippets or logs.

We appreciate your effort in keeping this learning repository secure and reliable.
