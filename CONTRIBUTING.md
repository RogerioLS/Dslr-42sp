# Contributing to 42 DSLR

## Code Standards
- All code must adhere to Python 3.10 standards.
- Lines must not exceed 100 characters (`black --line-length 100`, `flake8 --max-line-length=100`).
- Every module, function, and class must contain a complete docstring.
- Executable scripts must include an `if __name__ == '__main__':` guard.
- No prohibited built-in statistical functions in analytical code (`describe()`, `mean()`, `std()`, `min()`, `max()`, `percentile()`).

## Verification Workflow
Before submitting a pull request or pushing:
```bash
make audit
```
