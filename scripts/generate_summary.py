#!/usr/bin/env bash
"""Summary Generator for 42 DSLR GitHub Actions.

Runs incremental audit checks (syntax compilation, 42 norm & anti-cheating,
unit test discovery, security scan, data visualization plot artifacts) and produces:
- summary.md: Visual Markdown report for PR comments and $GITHUB_STEP_SUMMARY.
- artifacts/audit_summary.json: Metrics data for PR renamer and checklist updater.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

PLOT_DEFINITIONS: List[Dict[str, str]] = [
    {
        "filename": "histogram_grid.png",
        "cli": "histogram.py",
        "description": "Score distributions per Hogwarts house (homogeneous course identification)",
    },
    {
        "filename": "scatter_plot.png",
        "cli": "scatter_plot.py",
        "description": "Bivariate feature correlation & collinearity visualization",
    },
    {
        "filename": "pair_plot.png",
        "cli": "pair_plot.py",
        "description": "Multivariate feature pairwise relationship matrix",
    },
]


def run_command(cmd: list[str]) -> tuple[int, str]:
    """Runs a shell command and returns (exit_code, output_text)."""
    try:
        res = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True, check=False)
        output = (res.stdout + "\n" + res.stderr).strip()
        return res.returncode, output
    except Exception as e:
        return 1, str(e)


def _check_syntax() -> tuple[str, str, int]:
    """Compiles all existing Python files to check for syntax errors."""
    ignore_parts = (".venv", "venv", "__pycache__", ".git", "build", "dist")
    py_files = [
        str(p) for p in BASE_DIR.rglob("*.py") if not any(part in p.parts for part in ignore_parts)
    ]
    if not py_files:
        return "✅ PASSED", "No Python files found.", 0

    code, out = run_command([sys.executable, "-m", "py_compile", *py_files])
    status = "✅ PASSED" if code == 0 else "❌ FAILED"
    return status, out, len(py_files)


def _check_unit_tests() -> tuple[str, str, int, int]:
    """Runs existing unit tests and parses pass/total counts."""
    test_files = list((BASE_DIR / "tests").rglob("test_*.py"))
    if not test_files:
        return "✅ PASSED", "Ran 0 tests in 0.000s\n\nOK (Scaffolding stage)", 0, 0

    code, out = run_command(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
    )
    status = "✅ PASSED" if code == 0 else "❌ FAILED"

    total = 0
    passed = 0
    for line in out.splitlines():
        if "Ran " in line and "tests in" in line:
            try:
                total = int(line.split("Ran ")[1].split(" tests")[0])
                passed = total if code == 0 else 0
            except Exception:
                pass
    return status, out, passed, total


def _check_plot_artifacts() -> List[Dict[str, str]]:
    """Inspects workspace for generated data visualization PNG artifacts."""
    results: List[Dict[str, str]] = []
    for plot in PLOT_DEFINITIONS:
        file_path = BASE_DIR / plot["filename"]
        alt_path = ARTIFACTS_DIR / plot["filename"]
        target = file_path if file_path.exists() else (alt_path if alt_path.exists() else None)

        if target and target.exists():
            size_kb = target.stat().st_size / 1024
            results.append(
                {
                    "filename": plot["filename"],
                    "cli": plot["cli"],
                    "status": "✅ Generated",
                    "size": f"{size_kb:.1f} KB",
                    "description": plot["description"],
                }
            )
        else:
            cli_exists = (BASE_DIR / plot["cli"]).exists()
            status = "⏳ Pending Generation" if cli_exists else "⏳ In Development"
            results.append(
                {
                    "filename": plot["filename"],
                    "cli": plot["cli"],
                    "status": status,
                    "size": "-",
                    "description": plot["description"],
                }
            )
    return results


def _build_markdown(
    overall_status: str,
    timestamp: str,
    implemented_count: int,
    compile_status: str,
    py_count: int,
    norm_status: str,
    norm_errors: int,
    test_status: str,
    passed_tests: int,
    total_tests: int,
    sec_status: str,
    deliverables: dict[str, bool],
    plots: List[Dict[str, str]],
    norm_out: str,
    test_out: str,
) -> str:
    """Builds the comprehensive markdown report text."""
    avatar = "https://raw.githubusercontent.com/RogerioLS/RogerioLS/main/foto_little.png"
    md = [
        "# 🧙‍♂️ 42 DSLR — Audit & Quality Gate Report",
        f"**Overall Status**: {overall_status}  ",
        f"**Execution Timestamp**: `{timestamp}`  ",
        f"**Deliverables Progress**: `{implemented_count}/6` core deliverables present\n",
        "## 📊 Summary Overview",
        "| Metric | Status | Details |",
        "| :--- | :--- | :--- |",
        f"| ⚡ **Python 3.10 Syntax** | {compile_status} | Verified {py_count} file(s) |",
        f"| 🛡️ **42 Norm & Anti-Cheating** | {norm_status} | {norm_errors} norm error(s) |",
        f"| 🧪 **Unit Test Suites** | {test_status} | {passed_tests}/{total_tests} test(s) |",
        f"| 🔒 **Security Audit (Bandit)** | {sec_status} | Codebase vulnerability scan |",
        "\n## 📦 Deliverables Status",
        "| Deliverable | Phase | Status |",
        "| :--- | :--- | :--- |",
    ]

    phase_map = {
        "describe.py": "Phase 1: Stats",
        "histogram.py": "Phase 2: Viz",
        "scatter_plot.py": "Phase 2: Viz",
        "pair_plot.py": "Phase 2: Viz",
        "logreg_train.py": "Phase 3: Model",
        "logreg_predict.py": "Phase 3: Model",
    }
    for file_name, phase in phase_map.items():
        st = "✅ Ready" if deliverables[file_name] else "⏳ In Progress"
        md.append(f"| `{file_name}` | {phase} | {st} |")

    md.extend(
        [
            "\n## 📈 Data Visualizations & Plot Artifacts",
            "| Plot Artifact | Associated CLI | Status | File Size | Details |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
    )

    for p in plots:
        desc = p["description"]
        row = f"| `{p['filename']}` | `{p['cli']}` | " f"{p['status']} | {p['size']} | {desc} |"
        md.append(row)

    generated_plots = [p for p in plots if p["status"] == "✅ Generated"]
    if generated_plots:
        md.append("\n### 🖼️ Rendered Visual Artifacts (GitHub Pages CDN)\n")
        for p in generated_plots:
            img_url = f"https://rogeriols.github.io/Dslr-42sp/assets/plots/{p['filename']}"
            md.append(f"<details open><summary><b>📊 {p['filename']} Preview</b></summary>\n")
            md.append(
                f'<p align="center"><img src="{img_url}" alt="{p["filename"]}" width="90%"></p>\n'
            )
            md.append("</details>\n")

    md.extend(
        [
            "\n## 🔍 Audit Details\n",
            "<details><summary><b>View 42 Norm & Anti-Cheating Output</b></summary>\n",
            "```text",
            norm_out,
            "```",
            "</details>\n",
            "<details><summary><b>View Unit Test Execution Log</b></summary>\n",
            "```text",
            test_out,
            "```",
            "</details>\n",
            "---\n*Automated audit report generated for 42 DSLR.* "
            f'<img align="right" src="{avatar}" width="50">',
        ]
    )
    return "\n".join(md)


def main() -> None:
    """Executes the audit suite and generates markdown/json reports."""
    compile_status, _, py_count = _check_syntax()

    norm_code, norm_out = run_command([sys.executable, "scripts/norm_check.py"])
    norm_status = "✅ PASSED" if norm_code == 0 else "❌ FAILED"
    norm_errors = 0 if norm_code == 0 else 1

    test_status, test_out, passed_tests, total_tests = _check_unit_tests()

    sec_code, _ = run_command([sys.executable, "-m", "bandit", "-r", "src", "scripts", "-q"])
    sec_status = "✅ PASSED" if sec_code == 0 else "⚠️ REVIEW"

    deliverables = {
        name: (BASE_DIR / name).exists()
        for name in [
            "describe.py",
            "histogram.py",
            "scatter_plot.py",
            "pair_plot.py",
            "logreg_train.py",
            "logreg_predict.py",
        ]
    }
    implemented_count = sum(1 for v in deliverables.values() if v)

    plots = _check_plot_artifacts()

    overall_passed = (
        compile_status == "✅ PASSED" and norm_status == "✅ PASSED" and test_status == "✅ PASSED"
    )
    overall_status = "✅ AUDIT 100% PASSED" if overall_passed else "⚠️ AUDIT FAILED"
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    summary_text = _build_markdown(
        overall_status,
        timestamp,
        implemented_count,
        compile_status,
        py_count,
        norm_status,
        norm_errors,
        test_status,
        passed_tests,
        total_tests,
        sec_status,
        deliverables,
        plots,
        norm_out,
        test_out,
    )

    with open(BASE_DIR / "summary.md", "w", encoding="utf-8") as f:
        f.write(summary_text)

    metrics = {
        "timestamp": timestamp,
        "overall_passed": overall_passed,
        "implemented_count": implemented_count,
        "total_deliverables": 6,
        "py_files_count": py_count,
        "norm_errors": norm_errors,
        "passed_tests": passed_tests,
        "total_tests": total_tests,
        "plots": plots,
    }

    with open(ARTIFACTS_DIR / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("✔ Audit summary and metrics JSON generated successfully.")


if __name__ == "__main__":
    main()
