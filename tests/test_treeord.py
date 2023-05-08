from triglav import Triglav, ETCProx, CLRTransformer, NoScale, Scaler, NoResample

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

from skbio.stats.composition import clr, closure, multiplicative_replacement

import pandas as pd

import numpy as np

from pathlib import Path

# Checks for symmetry
def check_symmetric(X):

    if np.allclose(X, X.T, rtol = 1e-08, atol = 1e-08):
        pass

    else:
        raise ValueError("The matrix returned by ETCProx() is not symmetric")

dirpath = Path(__file__).parent

expected_output = dirpath / 'data/expected_output.csv'

# Tests the transformer and proximity modules
def test_transformers_prox():

    # Create the dataset
    X, y = make_classification(n_samples = 200,
                                n_features = 20,
                                n_informative = 5,
                                n_redundant = 2,
                                n_repeated = 0,
                                n_classes = 2,
                                shuffle = False,
                                random_state = 0)

    # To prevent negative proportions
    X = np.abs(X)

    # Ensures that the NoScale transformer returns the input
    R = NoScale().fit_transform(X)

    X = pd.DataFrame(X)
    R = pd.DataFrame(R)
    pd.testing.assert_frame_equal(X, R, check_dtype = False)

    # Ensures that CLRTransformer returns the CLR Transform of X
    R = pd.DataFrame(CLRTransformer().fit_transform(X))

    X_clr = pd.DataFrame(clr(multiplicative_replacement(closure(X))))
    pd.testing.assert_frame_equal(X_clr, R, check_dtype = False)

    # Ensures that Scaler returns the closure of X
    R = pd.DataFrame(Scaler().fit_transform(X))

    X_closure = pd.DataFrame(closure(X))
    pd.testing.assert_frame_equal(X_closure, R, check_dtype = False)

    # Ensures that NoResample returns the input
    R = pd.DataFrame(NoResample().fit_transform(X))

    pd.testing.assert_frame_equal(X, R, check_dtype = False)

    # Ensure that ETCProx returns a square matrix and symmetric matrixo
    R = ETCProx().transform(X)

    assert R.shape[0] == R.shape[1]
    check_symmetric(R)

    print("Transformer and Proximity Tests Complete.")


# Tests the overall Triglav pipeline
def test_triglav_basic():

    #Create the dataset
    X, y = make_classification(n_samples = 200,
                                n_features = 20,
                                n_informative = 5,
                                n_redundant = 2,
                                n_repeated = 0,
                                n_classes = 2,
                                shuffle = False,
                                random_state = 0)

    #Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.2, 
                                                        random_state = 0, 
                                                        stratify = y)

    #Set up Triglav
    model = Triglav(n_jobs = 5,
                    verbose = 3,
                    estimator = ExtraTreesClassifier(512, bootstrap = True, max_depth = 7),
                    metric = "euclidean",
                    linkage = "ward", 
                    criterion="maxclust",
                    thresh = 9,
                    transformer=StandardScaler())

    #Identify predictive features
    model.fit(X_train, y_train)

    features_selected = model.selected_
    features_best = model.selected_best_

    df = pd.DataFrame(data = [features_best, features_selected], index = ["Selected Best", "Selected"], columns = [str(i) for i in range(0, 20)])
    
    test_df = pd.read_csv(expected_output, index_col = 0)

    pd.testing.assert_frame_equal(df, test_df, check_dtype = False)

    print("Triglav Test Complete.")


    

