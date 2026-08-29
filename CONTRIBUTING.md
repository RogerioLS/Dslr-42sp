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

## 🏷️ Branch, Commit & Task Naming Governance (Strict Enforcement)

To ensure full traceability between the **GitHub Kanban**, **Pull Requests**, and **Git History**, all branches and commit messages are strictly validated by automated hooks and GitHub Actions.

### 🌿 1. Branch Naming Format:
```text
<type>/<task-id>-<short-description-in-kebab-case>
```
- **Valid Examples**:
  - `feat/dslr-01-data-pipeline`
  - `feat/dslr-02-stats-math`
  - `fix/dslr-03-percentile-interpolation`
  - `docs/dslr-13-peer-defense-guide`
  - `chore/infra-makefile-update`
- **Rejection Behavior**: If a branch name is invalid (e.g. `my-branch`, `test`, `dev`), the **GitHub Action (`branch_lint.yml`) will fail and block the Pull Request**.

---

### 📝 2. Commit Message Format:
```text
<type>(<scope>): [<TASK-ID>:#<ISSUE_NUM>] <short description in lowercase>
```
- **Valid Task Examples**:
  - `feat(analytics): [DSLR-01:#1] load csv and handle missing values`
  - `feat(math): [DSLR-02:#2] implement count and mean from scratch`
  - `feat(stats): [DSLR-03:#3] calculate 25th 50th 75th percentiles`
  - `docs(theory): [DSLR-09:#9] derive binary cross-entropy loss function`
- **Valid Non-Task / Infrastructure Examples**:
  - `chore(build): [INFRA] configure pre-commit hooks and make check`
  - `docs(meta): [DOCS] update contributing guidelines and security policy`
  - `fix(types): [HOTFIX] resolve lint typing issue in script loader`

### 📋 Allowed Reserved Tags (for non-subject changes):
`[INFRA]`, `[CHORE]`, `[DOCS]`, `[FIX]`, `[HOTFIX]`, `[SECURITY]`, `[GLOBAL]`, `[CONFIG]`, `[DEPS]`

---

### 🔄 3. Dynamic Task Lifecycle & Auto-Sync:
1. **Local Dynamic Detection**: The `.githooks/commit-msg` dynamically checks `.github/issues/`. When a new file `dslr-14-bonus.md` is added, the `[DSLR-14]` tag is **immediately valid** without editing any configuration!
2. **GitHub Web UI Sync**: If you or your peer open a new Issue on GitHub Web, simply run:
   ```bash
   make sync-tasks
   ```
   This downloads the new issue into `.github/issues/` so local git hooks recognize it offline.

---

### 🚨 4. Troubleshooting: What if my Commit or PR is Rejected?

- **If your commit was rejected by Git Hook**:
  ```text
  ⛔ COMMIT REJEITADO: TASK NÃO ENCONTRADA NO PROJETO
  ```
  1. Check if the task ID matches an existing file in `.github/issues/` (e.g. `[DSLR-01]` to `[DSLR-13]`);
  2. If it is a new task, create `.github/issues/dslr-XX-title.md`;
  3. If it is a general improvement without a subject task, use `[INFRA]` or `[CHORE]`.

- **If your PR was rejected by Branch Lint Action**:
  Rename your local branch and update the remote:
  ```bash
  git branch -m <old-name> feat/<task-id>-<description>
  git push origin -u feat/<task-id>-<description>
  git push origin --delete <old-name>
  ```

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
