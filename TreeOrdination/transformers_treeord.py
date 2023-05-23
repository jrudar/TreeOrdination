import numpy as np

from skbio.stats.composition import multiplicative_replacement, closure, clr

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import resample


class CLRClosureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, do_clr=False, delta=None):

        self.do_clr = do_clr
        self.delta = delta

    def fit_transform(self, X, y=None, **kwargs):

        if self.do_clr:
            return clr(multiplicative_replacement(closure(X[rows_to_keep]), delta=self.delta))

        else:
            return closure(X)

    def transform(self, X, y=None, **kwargs):

        if self.do_clr:
            return clr(multiplicative_replacement(closure(X[rows_to_keep]), delta=self.delta))

        else:
            return closure(X)


class ResampleRandomizeTransform(BaseEstimator, TransformerMixin):
    def __init__(self, resampler, transformer, exclude_cols):

        self.resampler = resampler
        self.transformer = transformer
        self.exclude_cols = exclude_cols

    def fit_resample(self, X, y):
        
        # Resample data if a resampler is supplied
        if isinstance(self.resampler, type(None)):
            X_re = np.copy(X)

        else:
            self.resampler = clone(self.resampler)

            X_re, _ = self.resampler.fit_resample(X, y)

        y_re = np.asarray([0 for _ in range(X_re.shape[0])])

        # Create Randomized Data
        X_perm = np.copy(X_re, "C")
        for col in range(X_perm.shape[1]):
            X_perm[:, col] = np.random.choice(
                X_perm[:, col], replace=False, size=X_perm.shape[0]
            )

        y_ra = np.asarray([1 for _ in range(X_perm.shape[0])])

        # Combine the real and permuted data
        X_comb = np.vstack((X_re, X_perm))

        y_comb = np.hstack((y_re, y_ra))

        # Prepare dataset if columns are excluded from transformation
        if self.exclude_cols[0]:
            # Split data into columns excluded from transformation and included for transformation
            excl_range = np.asarray(self.exclude_cols[1])

            X_transform = np.delete(X_comb, excl_range, axis=1)
            X_no_transform = X_comb[:, excl_range]

            # Remove zero-variance features
            self.zero_var_transformer = VarianceThreshold().fit(X_transform)
            self.zero_var_no_transformer = VarianceThreshold().fit(X_no_transform)

            X_transform = self.zero_var_transformer.transform(X_transform)
            X_no_transform = self.zero_var_no_transformer.transform(X_no_transform)

            # Transform data if a transformer is supplied
            if isinstance(self.transformer, type(None)) == False:
                self.transformer = clone(self.transformer)

                X_transform = self.transformer.fit_transform(X_transform)

            # Recombine
            X_comb = np.hstack((X_transform, X_no_transform))

            return X_comb, y_comb

        # Prepare dataset if columns are excluded from transformation
        else:
            # Remove zero-variance features
            self.zero_var_transformer = VarianceThreshold().fit(X_comb)
            X_transform = self.zero_var_transformer.transform(X_comb)

            # Transform data if a transformer is supplied
            if isinstance(self.transformer, type(None)) == False:
                self.transformer = clone(self.transformer)

                # Transform data
                X_transform = self.transformer.fit_transform(X_transform)

            return X_transform, y_comb

    def transform(self, X):

        # Prepare dataset if columns are excluded from transformation
        if self.exclude_cols[0]:
            # Split data into columns excluded from transformation and included for transformation
            excl_range = np.asarray(self.exclude_cols[1])

            X_transform = np.delete(X, excl_range, axis=1)
            X_no_transform = X[:, excl_range]

            # Remove zero variance features
            X_transform = self.zero_var_transformer.transform(X_transform)
            X_no_transform = self.zero_var_no_transformer.transform(X_no_transform)

            # Transform data if a transformer is supplied
            if isinstance(self.transformer, type(None)) == False:
                X_transform = self.transformer.transform(X_transform)

            if X_transform.ndim == 1:
                X_transform = [X_transform]

            # Recombine
            return np.hstack((X_transform, X_no_transform))

        else:
            # Remove zero-variance features
            X_transform = self.zero_var_transformer.transform(X)

            # Transform data if a transformer is supplied
            if isinstance(self.transformer, type(None)) == False:
                X_transform = self.transformer.transform(X_transform)

            return X_transform


