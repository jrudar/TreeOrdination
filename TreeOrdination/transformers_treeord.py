import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import resample


def closure(mat):
    """Perform closure to ensure that all elements add up to 1.

    From: https://github.com/scikit-bio/scikit-bio/blob/main/skbio/stats/composition.py

    Parameters
    ----------
    mat : array_like of shape (n_compositions, n_components)
        A matrix of proportions.

    Returns
    -------
    ndarray of shape (n_compositions, n_components)
        The matrix where all of the values are non-zero and each composition
        (row) adds up to 1.

    Raises
    ------
    ValueError
        If any values are negative.
    ValueError
        If the matrix has more than two dimensions.
    ValueError
        If there is a row that has all zeros.

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import closure
    >>> X = np.array([[2, 2, 6], [4, 4, 2]])
    >>> closure(X)
    array([[ 0.2,  0.2,  0.6],
           [ 0.4,  0.4,  0.2]])

    """
    mat = np.atleast_2d(mat)
    if np.any(mat < 0):
        raise ValueError("Cannot have negative proportions")
    if mat.ndim > 2:
        raise ValueError("Input matrix can only have two dimensions or less")
    if np.all(mat == 0, axis=1).sum() > 0:
        raise ValueError("Input matrix cannot have rows with all zeros")
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat.squeeze()


def multiplicative_replacement(mat, delta=None):
    r"""Replace all zeros with small non-zero values.

    It uses the multiplicative replacement strategy [1]_, replacing zeros with
    a small positive :math:`\delta` and ensuring that the compositions still
    add up to 1.

    From: https://github.com/scikit-bio/scikit-bio/blob/main/skbio/stats/composition.py

    Parameters
    ----------
    mat : array_like of shape (n_compositions, n_components)
        A matrix of proportions.
    delta : float, optional
        A small number to be used to replace zeros. If not specified, the
        default value is :math:`\delta = \frac{1}{N^2}` where :math:`N` is the
        number of components.

    Returns
    -------
    ndarray of shape (n_compositions, n_components)
        The matrix where all of the values are non-zero and each composition
        (row) adds up to 1.

    Raises
    ------
    ValueError
        If negative proportions are created due to a large ``delta``.

    Notes
    -----
    This method will result in negative proportions if a large delta is chosen.

    References
    ----------
    .. [1] J. A. Martin-Fernandez. "Dealing With Zeros and Missing Values in
           Compositional Data Sets Using Nonparametric Imputation"

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import multi_replace
    >>> X = np.array([[.2, .4, .4, 0],[0, .5, .5, 0]])
    >>> multi_replace(X)
    array([[ 0.1875,  0.375 ,  0.375 ,  0.0625],
           [ 0.0625,  0.4375,  0.4375,  0.0625]])

    """
    mat = closure(mat)
    z_mat = mat == 0

    num_feats = mat.shape[-1]
    tot = z_mat.sum(axis=-1, keepdims=True)

    if delta is None:
        delta = (1.0 / num_feats) ** 2

    zcnts = 1 - tot * delta
    if np.any(zcnts) < 0:
        raise ValueError(
            "The multiplicative replacement created negative "
            "proportions. Consider using a smaller `delta`."
        )
    mat = np.where(z_mat, delta, zcnts * mat)
    return mat.squeeze()


def clr(mat):
    r"""Perform centre log ratio transformation.

    From: https://github.com/scikit-bio/scikit-bio/blob/main/skbio/stats/composition.py

    This function transforms compositions from Aitchison geometry to the real
    space. The :math:`clr` transform is both an isometry and an isomorphism
    defined on the following spaces:

    .. math::
        clr: S^D \rightarrow U

    where :math:`U=
    \{x :\sum\limits_{i=1}^D x = 0 \; \forall x \in \mathbb{R}^D\}`

    It is defined for a composition :math:`x` as follows:

    .. math::
        clr(x) = \ln\left[\frac{x_1}{g_m(x)}, \ldots, \frac{x_D}{g_m(x)}\right]

    where :math:`g_m(x) = (\prod\limits_{i=1}^{D} x_i)^{1/D}` is the geometric
    mean of :math:`x`.

    Parameters
    ----------
    mat : array_like of shape (n_compositions, n_components)
        A matrix of proportions.

    Returns
    -------
    ndarray of shape (n_compositions, n_components)
        Clr-transformed matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from skbio.stats.composition import clr
    >>> x = np.array([.1, .3, .4, .2])
    >>> clr(x)
    array([-0.79451346,  0.30409883,  0.5917809 , -0.10136628])

    """
    mat = closure(mat)
    lmat = np.log(mat)
    gm = lmat.mean(axis=-1, keepdims=True)
    return (lmat - gm).squeeze()


class CLRClosureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, do_clr=False, delta=None):

        self.do_clr = do_clr
        self.delta = delta

    def fit_transform(self, X, y=None, **kwargs):

        if self.do_clr:
            return clr(multiplicative_replacement(closure(X), delta=self.delta)).astype(np.float32)

        else:
            return closure(X).astype(np.float32)

    def transform(self, X, y=None, **kwargs):

        if self.do_clr:
            return clr(multiplicative_replacement(closure(X), delta=self.delta)).astype(np.float32)

        else:
            return closure(X).astype(np.float32)


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

            if X_transform.ndim == 1:
                X_transform = np.asarray([X_transform])

            return X_transform


