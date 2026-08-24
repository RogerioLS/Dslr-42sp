# Anti-Cheating & Integrity Rules

The 42 subject explicitly states:
> "It is forbidden to use any function that does the job for you, such as: count, mean, std, min, max, percentile, etc., no matter the language that you use. Of course, it is also forbidden to use the describe library or any function that looks (more or less) similar to it from another library."

### Banned in `describe.py` and `src/analytics/`:
- `df.describe()`, `df.mean()`, `df.std()`, `df.min()`, `df.max()`, `df.count()`, `df.quantile()`
- `np.mean()`, `np.std()`, `np.var()`, `np.median()`, `np.percentile()`, `np.quantile()`
- `scipy.stats` aggregates
- `sklearn.linear_model.LogisticRegression` in the mandatory project deliverables

### Permitted:
- `pandas` for reading CSV into tabular structure (`pd.read_csv`) and indexing.
- `numpy` for raw array representations, vector dot products (`np.dot`), exponentials (`np.exp`) in mathematical formulas.
- `matplotlib` / `seaborn` for generating visualization figures.
- `sklearn.metrics.accuracy_score` in tests/evaluation verification scripts only.
