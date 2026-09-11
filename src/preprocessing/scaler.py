"""Handcrafted Z-Score Feature Standardization (StandardScaler) for 42 DSLR.

Normalizes feature distributions to zero mean (mu = 0) and unit variance (sigma = 1),
ensuring circular/spherical loss contours for stable and rapid Gradient Descent
convergence without numerical overflow.

Adheres strictly to the 42 Norm and Anti-Cheating protocols: all statistical moments
and scaling transformations are computed from first principles without external ML libraries.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from src.analytics.statistics import compute_mean, compute_std


class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample x is calculated as:
        z = (x - mu) / sigma

    where mu is the mean of the training samples and sigma is the standard deviation.
    """

    def __init__(
        self,
        with_mean: bool = True,
        with_std: bool = True,
        impute_strategy: str = "mean",
    ) -> None:
        """Initializes the uncalibrated StandardScaler.

        Args:
            with_mean (bool): If True, center data before scaling. Defaults to True.
            with_std (bool): If True, scale data to unit variance. Defaults to True.
            impute_strategy (str): Strategy for missing NaN values ('mean' or 'none').
                'mean': Replaces NaNs with the training feature mean (maps to z=0.0).
                'none': Preserves NaN values in output. Defaults to 'mean'.
        """
        self.with_mean = with_mean
        self.with_std = with_std
        self.impute_strategy = impute_strategy

        self.mean_: Dict[str, float] = {}
        self.std_: Dict[str, float] = {}
        self.impute_values_: Dict[str, float] = {}
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def fit(
        self,
        X: Union[pd.DataFrame, Sequence[Sequence[float]], Dict[str, Sequence[float]]],
        columns: Optional[Sequence[str]] = None,
    ) -> "StandardScaler":
        """Computes the mean and standard deviation for each feature from training data.

        Args:
            X: Input training dataset (DataFrame, 2D Sequence, or Dict of columns).
            columns: Optional subset of column names to calibrate on.

        Returns:
            StandardScaler: The calibrated scaler instance.

        Raises:
            ValueError: If the dataset contains no valid observations or is empty.
        """
        features_dict = self._to_features_dict(X, columns)
        if not features_dict:
            raise ValueError("Cannot fit StandardScaler on empty dataset.")

        self.mean_ = {}
        self.std_ = {}
        self.impute_values_ = {}
        self.feature_names_ = list(features_dict.keys())

        for col, raw_vals in features_dict.items():
            valid_vals = [float(v) for v in raw_vals if pd.notna(v)]
            if not valid_vals:
                raise ValueError(f"Feature '{col}' contains no valid numerical values.")

            mu = compute_mean(valid_vals)
            sigma = compute_std(valid_vals) if len(valid_vals) > 1 else 0.0

            self.mean_[col] = mu if self.with_mean else 0.0
            self.std_[col] = sigma if self.with_std else 1.0
            self.impute_values_[col] = mu

        self.is_fitted_ = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, Sequence[Sequence[float]], Dict[str, Sequence[float]]],
    ) -> Union[pd.DataFrame, List[List[float]]]:
        """Standardizes features using the stored training statistics (zero data leakage).

        Args:
            X: Input dataset to transform.

        Returns:
            Union[pd.DataFrame, List[List[float]]]: Standardized features in the same
            structural format as input.

        Raises:
            RuntimeError: If transform is called prior to fitting the scaler.
            KeyError: If a required calibrated feature is missing in input data.
        """
        if not self.is_fitted_:
            raise RuntimeError("StandardScaler must be fitted before transforming data.")

        is_df = isinstance(X, pd.DataFrame)
        features_dict = self._to_features_dict(X, self.feature_names_)

        scaled_dict: Dict[str, List[float]] = {}
        for col in self.feature_names_:
            if col not in features_dict:
                raise KeyError(
                    f"Feature '{col}' calibrated during fit is missing in transform data."
                )

            raw_vals = features_dict[col]
            mu = self.mean_[col]
            sigma = self.std_[col]

            # Epsilon protection against constant feature division by zero
            scale_divisor = sigma if sigma > 1e-12 else 1.0

            col_scaled: List[float] = []
            for val in raw_vals:
                if pd.isna(val):
                    if self.impute_strategy == "mean":
                        # Standardizing the mean yields exact zero: (mu - mu) / sigma = 0.0
                        col_scaled.append(0.0)
                    else:
                        col_scaled.append(float("nan"))
                else:
                    z = (float(val) - mu) / scale_divisor
                    col_scaled.append(z)

            scaled_dict[col] = col_scaled

        if is_df:
            # Return new DataFrame preserving index and non-scaled columns if partial
            result_df = X.copy()
            for col, vals in scaled_dict.items():
                result_df[col] = vals
            return result_df

        # If 2D matrix or dictionary, return structured row-wise list of lists
        num_rows = len(next(iter(scaled_dict.values())))
        matrix: List[List[float]] = []
        for i in range(num_rows):
            row = [scaled_dict[col][i] for col in self.feature_names_]
            matrix.append(row)
        return matrix

    def fit_transform(
        self,
        X: Union[pd.DataFrame, Sequence[Sequence[float]], Dict[str, Sequence[float]]],
        columns: Optional[Sequence[str]] = None,
    ) -> Union[pd.DataFrame, List[List[float]]]:
        """Fits to data, then transforms it in an atomic operation.

        Args:
            X: Input dataset to fit and transform.
            columns: Optional subset of column names to scale.

        Returns:
            Union[pd.DataFrame, List[List[float]]]: Standardized dataset.
        """
        return self.fit(X, columns).transform(X)

    def inverse_transform(
        self,
        X_scaled: Union[pd.DataFrame, Sequence[Sequence[float]], Dict[str, Sequence[float]]],
    ) -> Union[pd.DataFrame, List[List[float]]]:
        """Unscales standardized data back to the original feature domain.

        Formula:
            x_original = z * sigma + mu

        Args:
            X_scaled: Standardized dataset to invert.

        Returns:
            Union[pd.DataFrame, List[List[float]]]: Inverted data in original domain.

        Raises:
            RuntimeError: If called before scaler is fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError("StandardScaler must be fitted before inverse transforming.")

        is_df = isinstance(X_scaled, pd.DataFrame)
        features_dict = self._to_features_dict(X_scaled, self.feature_names_)

        unscaled_dict: Dict[str, List[float]] = {}
        for col in self.feature_names_:
            raw_vals = features_dict[col]
            mu = self.mean_[col]
            sigma = self.std_[col]
            scale_factor = sigma if sigma > 1e-12 else 0.0

            col_unscaled: List[float] = []
            for val in raw_vals:
                if pd.isna(val):
                    col_unscaled.append(float("nan"))
                else:
                    x = float(val) * scale_factor + mu
                    col_unscaled.append(x)
            unscaled_dict[col] = col_unscaled

        if is_df:
            result_df = X_scaled.copy()
            for col, vals in unscaled_dict.items():
                result_df[col] = vals
            return result_df

        num_rows = len(next(iter(unscaled_dict.values())))
        matrix: List[List[float]] = []
        for i in range(num_rows):
            row = [unscaled_dict[col][i] for col in self.feature_names_]
            matrix.append(row)
        return matrix

    def _to_features_dict(
        self,
        data: Any,
        columns: Optional[Sequence[str]] = None,
    ) -> Dict[str, List[Any]]:
        """Normalizes heterogeneous input structures into a column-oriented dictionary.

        Args:
            data: Input data (DataFrame, 2D Sequence, or Dict).
            columns: Selected column names.

        Returns:
            Dict[str, List[Any]]: Dictionary mapping feature names to value lists.
        """
        if isinstance(data, pd.DataFrame):
            target_cols = list(columns) if columns else list(data.columns)
            return {col: list(data[col]) for col in target_cols if col in data.columns}

        if isinstance(data, dict):
            target_cols = list(columns) if columns else list(data.keys())
            return {col: list(data[col]) for col in target_cols if col in data}

        # Handle 2D list/matrix: [[x1, y1], [x2, y2], ...]
        if isinstance(data, (list, tuple)):
            if not data:
                return {}
            first_row = data[0]
            if not isinstance(first_row, (list, tuple)):
                # 1D single sequence
                col_name = columns[0] if columns else "feat_0"
                return {col_name: list(data)}

            num_cols = len(first_row)
            col_names = (
                list(columns)
                if columns and len(columns) == num_cols
                else [f"feat_{j}" for j in range(num_cols)]
            )
            return {col_names[j]: [row[j] for row in data] for j in range(num_cols)}

        return {}

    def to_dict(self) -> Dict[str, Any]:
        """Serializes fitted scaler parameters to a dictionary.

        Returns:
            Dict[str, Any]: Scaler metadata and statistics.
        """
        return {
            "with_mean": self.with_mean,
            "with_std": self.with_std,
            "impute_strategy": self.impute_strategy,
            "mean_": self.mean_,
            "std_": self.std_,
            "impute_values_": self.impute_values_,
            "feature_names_": self.feature_names_,
            "is_fitted_": self.is_fitted_,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardScaler":
        """Reconstructs a fitted StandardScaler from a serialized dictionary.

        Args:
            data (Dict[str, Any]): Dictionary containing scaler parameters.

        Returns:
            StandardScaler: Reconstructed scaler instance.
        """
        scaler = cls(
            with_mean=data.get("with_mean", True),
            with_std=data.get("with_std", True),
            impute_strategy=data.get("impute_strategy", "mean"),
        )
        scaler.mean_ = data.get("mean_", {})
        scaler.std_ = data.get("std_", {})
        scaler.impute_values_ = data.get("impute_values_", {})
        scaler.feature_names_ = data.get("feature_names_", [])
        scaler.is_fitted_ = data.get("is_fitted_", False)
        return scaler
