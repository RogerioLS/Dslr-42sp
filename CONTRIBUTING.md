# Contributing Guidelines — 42 DSLR

Welcome to the **42 DSLR (Data Science × Logistic Regression)** repository! We follow rigorous software engineering, academic integrity, and clean code standards.

---

## 🛠️ Centralized Workflow with Makefile

We use an interactive **Makefile Command Center** to standardize all development, linting, testing, and execution tasks. Always prefer using `make` commands rather than invoking raw python commands:

| Command | Purpose | When to Use |
| :--- | :--- | :--- |
| `make help` | Displays the interactive command menu with descriptions | Anytime you need a quick command refresher |
| `make install` | Installs project dependencies in your local virtual environment | First-time project setup |
| `make check` | Runs full pre-commit linters and AST norm checks | **Before staging files or creating commits** |
| `make norm` | Audits docstrings, `__main__` guards, and AST anti-cheating rules | To verify strict 42 norm compliance |
| `make compile` | Verifies Python 3.10 syntax compilation across all files | Fast syntax validation |
| `make test` | Executes all unit and integration test suites recursively | After any code modification |
| `make audit` | Runs the full verification suite (`compile` + `norm` + `test`) | Before pushing or opening a Pull Request |
| `make summary` | Generates local Markdown audit report (`summary.md`) | To preview the PR audit comment locally |
| `make describe` | Runs `describe.py` on the training dataset | Milestone 1 verification |
| `make histogram`| Generates course distribution plots (`histogram.py`) | Milestone 2 feature exploration |
| `make scatter` | Generates feature correlation scatter plots (`scatter_plot.py`) | Milestone 2 feature exploration |
| `make pairplot` | Generates the complete 13x13 pair plot matrix (`pair_plot.py`) | Milestone 2 feature exploration |
| `make train` | Trains One-vs-Rest Logistic Regression & saves weights | Milestone 3 model training |
| `make predict` | Runs inference on test dataset & generates `houses.csv` | Milestone 3 model prediction |
| `make evaluate` | Evaluates classification accuracy against the $\ge 98\%$ threshold | Milestone 4 defense readiness |
| `make clean` | Removes temporary cache files (`__pycache__`, `.pyc`, etc.) | Workspace hygiene |

---

## 🌿 Git Branching Strategy

- `main`: Protected branch containing stable, fully audited code.
- `feat/<task-id>-<description>`: Feature branches for specific tasks (e.g. `feat/dslr-01-data-pipeline`).
- `fix/<issue-id>-<description>`: Bug fixes and refactoring.
- `docs/<topic>`: Documentation and mathematical derivations.

---

## 🔒 Commit Conventions (Conventional Commits)

Commit messages must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```text
<type>(<scope>): <short description in imperative mood>
```

### Types:
- `feat`: New feature, algorithm, or deliverable script (`describe.py`, `histogram.py`, `train.py`).
- `fix`: Bug fix or logic correction.
- `docs`: Documentation, mathematical proofs, or docstring additions.
- `test`: Adding or modifying unit / integration tests.
- `chore`: Tooling, linters, pre-commit, or CI/CD configuration.
- `refactor`: Code restructuring without changing external behavior.

---

## 🧪 Testing Pyramid Architecture

All tests must be placed in the `tests/` directory following this structure:
- `tests/unit/`: Tests individual mathematical functions in isolation (`src/analytics/`, `src/model/`).
- `tests/integration/`: Tests end-to-end CLI behavior (`describe.py`, `logreg_train.py`, `logreg_predict.py`).

Execute all test suites with:
```bash
make test
```

---

## 🛡️ 42 Norm & Academic Integrity Rules

1. **Python 3.10 Compatibility**: Strictly use Python 3.10 standard library and allowed dependencies.
2. **Line Length Limit**: Maximum **100 characters per line** enforced by Black, Flake8, and Ruff.
3. **Documentation**: Every module, class, and public function must have a complete docstring.
4. **Execution Guards**: All CLI scripts must be wrapped inside `if __name__ == "__main__":`.
5. **No Prohibited Built-in ML / Stats**:
   - `describe.py` must compute Count, Mean, Std, Min, 25%, 50%, 75%, Max from scratch.
   - Using `pandas.DataFrame.describe()`, `numpy.mean()`, `numpy.std()`, `scipy.stats` in analytics core is strictly forbidden.
   - Scikit-Learn models (`LogisticRegression`) are prohibited for the main model implementation.

---

## 🚀 Pre-Push Checklist

Before pushing commits or opening a PR, ensure:
1. `make check` passes with zero errors;
2. `make audit` executes with 100% tests passing and 0 norm violations;
3. `CHANGELOG.md` is updated with your changes under `[Unreleased]`.
