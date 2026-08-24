# Global Coding Rules

1. **Python Standard**: Target Python 3.10+.
2. **Formatting**: Maximum line length of 100 characters. Follow Black and Flake8 standards.
3. **Documentation**: Every function, method, class, and module must have a clear docstring (`__doc__`).
4. **Execution Guards**: Executable scripts must use the standard `if __name__ == "__main__":` entrypoint guard.
5. **No Pollution**: Never commit temporary cache files (`__pycache__`, `.pytest_cache`, `.DS_Store`).
6. **Explicit Errors**: Handle bad arguments, non-existent files, and invalid formats with informative error messages.
