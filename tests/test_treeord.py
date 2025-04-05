from TreeOrdination import (
    TreeOrdination,
    CLRClosureTransformer,
    ResampleRandomizeTransform
)

from TreeOrdination.transformers_treeord import (
    multiplicative_replacement,
    closure,
    clr
)

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, median_absolute_error

import pandas as pd

import numpy as np

from pathlib import Path

dirpath = Path(__file__).parent

# Tests the transformer modules
def test_transformers():

    # Create the dataset
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=5,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        shuffle=False,
        random_state=0,
    )

    # To prevent negative proportions
    X_non_neg = np.abs(X)

    # Ensures that CLRTransformer returns the CLR Transform of X
    R = pd.DataFrame(CLRClosureTransformer(do_clr=True).fit_transform(X_non_neg))

    X_clr = pd.DataFrame(clr(multiplicative_replacement(closure(X_non_neg))))
    pd.testing.assert_frame_equal(X_clr, R, check_dtype=False)

    # Ensures that CLRTransformer returns the Closure of X
    R = pd.DataFrame(CLRClosureTransformer(do_clr=False).fit_transform(X_non_neg))
    X_closure = pd.DataFrame(closure(X_non_neg))
    pd.testing.assert_frame_equal(X_closure, R, check_dtype=False)

    # Ensure sampler, transformer objected works
    R_trf = ResampleRandomizeTransform(None, StandardScaler(), [False, -1])
    R, y_random = R_trf.fit_resample(X, y)
    
    assert R.shape[0] == int(X.shape[0] * 2)
    assert R.shape[1] == X.shape[1]
    assert y_random.shape[0] == int(X.shape[0] * 2)

# Tests the overall TreeOrdination pipeline
def test_treeord_basic():

    # Create the dataset
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=5,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        shuffle=False,
        random_state=0,
    )

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    # Set up TreeOrdinaton
    model = TreeOrdination(
        feature_names=[i for i in range(0, X.shape[1])],
        transformer=StandardScaler(),
        n_iter_unsup=2,
    )

    # Identify predictive features
    model.fit(X_train, y_train)

    X_emb_train = model.emb_transform(X_train, "PCA")
    X_approx_train = model.emb_transform(X_train, "approx")

    X_emb_test = model.emb_transform(X_test, "PCA")
    X_approx_test = model.emb_transform(X_test, "approx")

    error_train = median_absolute_error(X_emb_train, X_approx_train)
    assert error_train <= 0.75

    error_test = median_absolute_error(X_emb_test, X_approx_test)
    assert error_test <= 0.75

    p = model.predict(X_test)
    s = balanced_accuracy_score(y_test, p)
    assert s >= 0.75

    model.plot_projection(X, y)

    model.get_importance()
    model.plot_importance_global(X_test, y_test, 0, 10, 0)
    model.plot_importance_local(X_test[8], 10, 0)

if __name__ == "__main__":

    test_transformers()
    test_treeord_basic()
