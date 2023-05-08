from TreeOrdination import TreeOrdination, CLRClosureTransformer, NoTransform, NoResample

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

import pandas as pd

import numpy as np

from pathlib import Path

dirpath = Path(__file__).parent

expected_output = dirpath / 'data/expected_output.csv'

# Tests the transformer modules
def test_transformers():

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
    X_non_neg = np.abs(X)

    # Ensures that the NoScale transformer returns the input
    R = NoTransform().fit_transform(X)

    X = pd.DataFrame(X)
    R = pd.DataFrame(R)
    pd.testing.assert_frame_equal(X, R, check_dtype = False)

    # Ensures that CLRTransformer returns the CLR Transform of X
    R = pd.DataFrame(CLRClosureTransformer(do_clr = True).fit_transform(X))

    X_clr = pd.DataFrame(clr(multiplicative_replacement(closure(X))))
    pd.testing.assert_frame_equal(X_clr, R, check_dtype = False)

    # Ensures that CLRTransformer returns the Closure of X
    R = pd.DataFrame(CLRClosureTransformer(do_clr = False).fit_transform(X))

    X_closure = pd.DataFrame(closure(X))
    pd.testing.assert_frame_equal(X_closure, R, check_dtype = False)

    # Ensures that NoResample returns the input
    R = pd.DataFrame(NoResample().fit_transform(X))

    pd.testing.assert_frame_equal(X, R, check_dtype = False)


# Tests the overall TreeOrdination pipeline
def test_treeord_basic():

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

    #Set up TreeOrdinaton
    model = TreeOrdination(feature_names = [i for i in range(0, X.shape[1])])

    #Identify predictive features
    model.fit(X_train, y_train)




    

