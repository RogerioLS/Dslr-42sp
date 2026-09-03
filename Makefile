# ==============================================================================
#                       42 DSLR MASTER MAKEFILE (COMMAND CENTER)
# ==============================================================================

PYTHON := python3
TRAIN_DATA := datasets/dataset_train.csv
TEST_DATA  := datasets/dataset_test.csv
WEIGHTS    := weights.json
PRED_FILE  := houses.csv

# ANSI Color Codes & Formatting
RESET   := \033[0m
BOLD    := \033[1m
DIM     := \033[2m
CYAN    := \033[36m
GREEN   := \033[32m
YELLOW  := \033[33m
RED     := \033[31m
MAGENTA := \033[35m
BLUE    := \033[34m
WHITE   := \033[97m

.PHONY: help install describe histogram scatter pairplot train predict evaluate test norm compile audit summary check pre-commit clean

help:
	@printf "$(CYAN)┌──────────────────────────────────────────────────────────────────────────────┐\n$(RESET)"
	@printf "$(CYAN)│$(RESET) $(BOLD)$(MAGENTA)                 42 DSLR — DATA SCIENCE COMMAND CENTER                      $(RESET) $(CYAN)│\n$(RESET)"
	@printf "$(CYAN)├──────────────────────────────────────────────────────────────────────────────┤\n$(RESET)"
	@printf "$(CYAN)│$(RESET)  $(BOLD)$(GREEN)make help$(RESET)       $(DIM)─$(RESET) Show this interactive help menu                           $(CYAN)│\n$(RESET)"
	@printf "$(CYAN)│$(RESET)  $(BOLD)$(GREEN)make install$(RESET)    $(DIM)─$(RESET) Install dependencies in local virtualenv / system         $(CYAN)│\n$(RESET)"
	@printf "$(CYAN)│$(RESET)  $(BOLD)$(GREEN)make describe$(RESET)   $(DIM)─$(RESET) Run describe.py on train dataset                          $(CYAN)│\n$(RESET)"
	@printf "$(CYAN)│$(RESET)  $(BOLD)$(GREEN)make histogram$(RESET)  $(DIM)─$(RESET) Run histogram.py (Course distribution analysis)           $(CYAN)│\n$(RESET)"
	@printf "$(CYAN)│$(RESET)  $(BOLD)$(GREEN)make scatter$(RESET)    $(DIM)─$(RESET) Run scatter_plot.py (Feature correlation analysis)        $(CYAN)│\n$(RESET)"
	@printf "$(CYAN)│$(RESET)  $(BOLD)$(GREEN)make pairplot$(RESET)   $(DIM)─$(RESET) Run pair_plot.py (Full pair plot matrix)                  $(CYAN)│\n$(RESET)"
	@printf "$(CYAN)│$(RESET)  $(BOLD)$(GREEN)make train$(RESET)      $(DIM)─$(RESET) Train One-vs-Rest Logistic Regression & save weights      $(CYAN)│\n$(RESET)"
	@printf "$(CYAN)│$(RESET)  $(BOLD)$(GREEN)make predict$(RESET)    $(DIM)─$(RESET) Predict houses for test dataset (generate houses.csv)     $(CYAN)│\n$(RESET)"
	@printf "$(CYAN)│$(RESET)  $(BOLD)$(GREEN)make evaluate$(RESET)   $(DIM)─$(RESET) Check test accuracy against 98%% threshold                 $(CYAN)│\n$(RESET)"
	@printf "$(CYAN)│$(RESET)  $(BOLD)$(GREEN)make test$(RESET)       $(DIM)─$(RESET) Run all unit test suites                                  $(CYAN)│\n$(RESET)"
	@printf "$(CYAN)│$(RESET)  $(BOLD)$(GREEN)make norm$(RESET)       $(DIM)─$(RESET) Run 42 Norm & Anti-Cheating Auditor                       $(CYAN)│\n$(RESET)"
	@printf "$(CYAN)│$(RESET)  $(BOLD)$(GREEN)make compile$(RESET)    $(DIM)─$(RESET) Compile Python 3.10 syntax across all project files       $(CYAN)│\n$(RESET)"
	@printf "$(CYAN)│$(RESET)  $(BOLD)$(GREEN)make audit$(RESET)      $(DIM)─$(RESET) Full audit: compile + norm + unit tests                   $(CYAN)│\n$(RESET)"
	@printf "$(CYAN)│$(RESET)  $(BOLD)$(GREEN)make summary$(RESET)    $(DIM)─$(RESET) Generate local audit summary report (summary.md)          $(CYAN)│\n$(RESET)"
	@printf "$(CYAN)│$(RESET)  $(BOLD)$(GREEN)make check$(RESET)      $(DIM)─$(RESET) Pre-commit sanity check across all project files          $(CYAN)│\n$(RESET)"
	@printf "$(CYAN)│$(RESET)  $(BOLD)$(GREEN)make pre-commit$(RESET) $(DIM)─$(RESET) Install pre-commit tool and set up git hooks              $(CYAN)│\n$(RESET)"
	@printf "$(CYAN)│$(RESET)  $(BOLD)$(GREEN)make clean$(RESET)      $(DIM)─$(RESET) Remove temporary cache and prediction files               $(CYAN)│\n$(RESET)"
	@printf "$(CYAN)├──────────────────────────────────────────────────────────────────────────────┤\n$(RESET)"
	@printf "$(CYAN)│$(RESET)           $(BOLD)$(WHITE)🔥 Crafted with • by $(YELLOW)@RogerioLS$(WHITE) $(DIM)•$(RESET) $(BOLD)$(CYAN)42 São Paulo 🇧🇷$(RESET)                  $(CYAN)│\n$(RESET)"
	@printf "$(CYAN)└──────────────────────────────────────────────────────────────────────────────┘\n$(RESET)"

install:
	@printf "$(BOLD)$(BLUE)📦 [INSTALL] Installing project dependencies and configuring git hooks...$(RESET)\n"
	@$(PYTHON) -m pip install -e ".[dev]"
	@bash scripts/install-hooks.sh
	@printf "$(GREEN)✔ Dependencies installed and Git hooks configured successfully!$(RESET)\n"

describe:
	@printf "$(BOLD)$(BLUE)📊 [DATA ANALYSIS] Running describe.py...$(RESET)\n"
	@$(PYTHON) describe.py $(TRAIN_DATA)

histogram:
	@printf "$(BOLD)$(BLUE)📈 [DATA VISUALIZATION] Running histogram.py...$(RESET)\n"
	@$(PYTHON) histogram.py $(TRAIN_DATA)

scatter:
	@printf "$(BOLD)$(BLUE)📈 [DATA VISUALIZATION] Running scatter_plot.py...$(RESET)\n"
	@$(PYTHON) scatter_plot.py $(TRAIN_DATA)

pairplot:
	@printf "$(BOLD)$(BLUE)📈 [DATA VISUALIZATION] Running pair_plot.py...$(RESET)\n"
	@$(PYTHON) pair_plot.py $(TRAIN_DATA)

train:
	@printf "$(BOLD)$(MAGENTA)🧠 [MODEL] Training Logistic Regression (One-vs-Rest)...$(RESET)\n"
	@$(PYTHON) logreg_train.py $(TRAIN_DATA)

predict:
	@printf "$(BOLD)$(MAGENTA)🔮 [MODEL] Predicting test set houses...$(RESET)\n"
	@$(PYTHON) logreg_predict.py $(TEST_DATA) $(WEIGHTS)

evaluate:
	@printf "$(BOLD)$(BLUE)🎯 [EVALUATION] Checking model accuracy...$(RESET)\n"
	@$(PYTHON) scripts/evaluate_accuracy.py

test:
	@printf "$(BOLD)$(BLUE)🚀 [TESTS] Running all unit test suites...$(RESET)\n"
	@$(PYTHON) -m unittest discover -s tests -p "test_*.py"

norm:
	@printf "$(BOLD)$(YELLOW)🛡️ [NORM] Running 42 Norm & Anti-Cheating Auditor...$(RESET)\n"
	@$(PYTHON) scripts/norm_check.py

compile:
	@printf "$(BOLD)$(MAGENTA)⚡ [COMPILE] Verifying Python 3.10 syntax compilation...$(RESET)\n"
	@$(PYTHON) -m py_compile $$(find src scripts tests -name "*.py" 2>/dev/null) $$(find . -maxdepth 1 -name "*.py")
	@printf "$(GREEN)✔ Syntax compilation successful!$(RESET)\n"

audit: compile norm test
	@printf "\n$(BOLD)$(GREEN)======================================================================$(RESET)\n"
	@printf "$(BOLD)$(GREEN)   ✅ FULL AUDIT COMPLETE: Code is compliant & ready for evaluation!   $(RESET)\n"
	@printf "$(BOLD)$(GREEN)======================================================================$(RESET)\n\n"

summary:
	@printf "$(BOLD)$(BLUE)📊 [SUMMARY] Generating local audit report (summary.md)...$(RESET)\n"
	@$(PYTHON) scripts/generate_summary.py

check:
	@printf "$(BOLD)$(YELLOW)🔍 [CHECK] Running full pre-commit validation across all files...$(RESET)\n"
	@$(PYTHON) scripts/norm_check.py
	@pre-commit run --all-files
	@printf "$(GREEN)✔ All pre-commit & norm checks passed! Ready for git commit.$(RESET)\n\n"

sync-tasks:
	@printf "$(BOLD)$(CYAN)🔄 [SYNC] Synchronizing GitHub issues to local task files...$(RESET)\n"
	@$(PYTHON) scripts/sync_tasks.py
	@printf "$(GREEN)✔ Tasks successfully synchronized!$(RESET)\n"



pre-commit:
	@if command -v pre-commit > /dev/null 2>&1; then \
		printf "$(GREEN)✔ pre-commit is already installed.$(RESET)\n"; \
	else \
		printf "$(YELLOW)⏳ Installing pre-commit via pip...$(RESET)\n"; \
		$(PYTHON) -m pip install pre-commit; \
	fi
	@pre-commit install > /dev/null 2>&1 || printf "$(YELLOW)ℹ Note: pre-commit hooks configured alongside custom git hooks.$(RESET)\n"
	@printf "$(GREEN)✔ pre-commit setup completed successfully!$(RESET)\n"

clean:
	@printf "$(BOLD)$(RED)🧹 [CLEAN] Removing temporary cache and build files...$(RESET)\n"
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name ".ruff_cache" -exec rm -rf {} +
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "houses.csv" -delete
	@find . -type f -name "weights.json" -delete
	@printf "$(GREEN)✔ Clean completed successfully.$(RESET)\n\n"
