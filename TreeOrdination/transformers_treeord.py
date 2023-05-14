from skbio.stats.composition import multiplicative_replacement, closure, clr
from sklearn.base import TransformerMixin, BaseEstimator


class NoTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit_transform(self, X, y=None, **kwargs):

        return X

    def transform(self, X, y=None, **kwargs):

        return X


class CLRClosureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, do_clr=False, delta=None):

        self.do_clr = do_clr
        self.delta = delta

    def fit_transform(self, X, y=None, **kwargs):

        if self.do_clr:
            return clr(multiplicative_replacement(closure(X), delta=self.delta))

        else:
            return closure(X)

    def transform(self, X, y=None, **kwargs):

        if self.do_clr:
            return clr(multiplicative_replacement(closure(X), delta=self.delta))

        else:
            return closure(X)


class NoResample(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit_resample(self, X, y, **kwargs):

        return X, y
