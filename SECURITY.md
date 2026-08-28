# Security Policy — 42 DSLR

## Overview

This repository contains the Machine Learning project for the **42 Specialization — DSLR (Data Science × Logistic Regression)**. We take code quality, safety, and responsible disclosure seriously.

---

## Supported Versions

Only the active Python version defined by the 42 curriculum is officially supported for security updates and exercise validation:

| Version | Supported |
| :--- | :---: |
| Python 3.10.x | :white_check_mark: Yes |
| Python < 3.10 | :x: No |

---

## 🛡️ Security Best Practices & Automated Scanning

Security and code integrity are enforced continuously through our automated pipeline and **Makefile** tooling:

1. **Automated Secret Detection (`make check`)**:
   - `detect-secrets` runs locally via pre-commit hooks before every commit.
   - Prevents accidental exposure of API keys, tokens, or private credentials.

2. **Static Security Analysis (`make audit`)**:
   - Bandit AST security scanning detects common Python vulnerabilities, unsafe des-serialization, and shell injection risks.

3. **Safe Input Validation & Robust Error Handling**:
   - CLI scripts sanitize dataset paths and input arguments.
   - Prevent arbitrary command execution and directory traversal attacks (`../`).
   - Gracefully handle exceptions without leaking raw stack traces or internal environment variables.

4. **Dependency Vulnerability Management**:
   - Dependabot regularly scans and submits automated PRs for Python dependencies (`pip`) and GitHub Actions.

5. **Environment Isolation**:
   - Always run the project inside an isolated virtual environment (`venv` or `conda`).
   - Avoid executing analytical scripts with elevated (root/sudo) privileges.

---

## 🚨 Reporting a Vulnerability

If you discover a security vulnerability or accidental credential exposure in this repository, please report it responsibly:

1. **Do NOT open a public GitHub issue.**
2. Submit a private security advisory via GitHub or contact the repository owner ([@RogerioLS](https://github.com/RogerioLS)).
3. Include detailed steps to reproduce the issue, along with relevant logs or code snippets.

We appreciate your effort in keeping this learning repository secure and reliable.
