import numpy as np

import pandas as pd

# For class construction
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import resample
from sklearn.decomposition import PCA

from umap import UMAP

#Randomization function
def addcl2(X):

    #Resample rows
    X_resamp = resample(X, replace = True, n_samples = X.shape[0])

    #Resample columns
    X_perm = np.copy(X_resamp, "C").transpose()
    for col in range(X_perm.shape[0]):
        X_perm[col] = resample(X_perm[col], replace = False, n_samples = X_perm.shape[1])
        
    #Create Labels
    y_new = [0 for _ in range(X_resamp.shape[0])]
    y_new.extend([1 for _ in range(X_resamp.shape[0])])
    y_new = np.asarray(y_new)
    
    #Create merged dataset
    X_new = np.vstack((X_resamp, X_perm.transpose()))
            
    return X_new, y_new

#Tree Ordination class
class TreeOrdination(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        
        feature_names,

        resample_on_y = False,

        metric = "jaccard",

        supervised_clf = ExtraTreesClassifier(1024),
        n_iter_unsup = 20,
        unsup_n_estim = 80,
        n_jobs = 4,

        n_neighbors = 8,
        n_components = 2,
        min_dist = 0.001,
        
    ):
        self.feature_names = feature_names

        self.resample_on_y = resample_on_y

        self.metric = metric

        self.supervised_clf = supervised_clf
        self.n_iter_unsup = n_iter_unsup
        self.unsup_n_estim = unsup_n_estim
        self.n_jobs = n_jobs

        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        
    def get_initial_embedding(self, X):

        #Get an Initial LANDMark Representation
        self.Rs = [LANDMarkClassifier(self.unsup_n_estim, use_nnet = False, n_jobs = self.n_jobs).fit(*addcl2(X)) for _ in range(self.n_iter_unsup)]

        self.feature_importances_ = np.mean([clf.feature_importances_ for clf in self.Rs], axis = 0)

        #Get Proximity
        self.R_final = np.hstack([R.proximity(X) for R in self.Rs])
        
        #Get Embeddings
        self.tree_emb = UMAP(n_neighbors = self.n_neighbors,
                             n_components = 15,
                             min_dist = self.min_dist,
                             metric = self.metric,
                             densmap = False
                             ).fit(self.R_final)

        self.R_PCA = PCA(self.n_components).fit(self.tree_emb.transform(self.R_final))

        self.R_PCA_emb = self.R_PCA.transform(self.tree_emb.transform(self.R_final))

    def fit(self, X, y = None):
        
        self.y = np.copy(y, "C")
        self.encoded_labels = LabelEncoder().fit(self.y)
        y_enc = self.encoded_labels.transform(self.y)

        #Get initial embeddings, features, and cluster
        self.f_name = np.asarray(self.feature_names)

        self.get_initial_embedding(X)

        #Train a Classification model
        self.p_model = clone(self.supervised_clf).fit(self.R_final, y_enc.astype(int))

        #Train a projection model
        self.l_model = clone(ExtraTreesRegressor(1024)).fit(X, self.R_PCA_emb)

        return self

    def predict_proba(self, X):

        tree_emb = np.hstack([R.proximity(X) for R in self.Rs])
        P = self.p_model.predict_proba(tree_emb)

        return P

    def predict(self, X):

        tree_emb = np.hstack([R.proximity(X) for R in self.Rs])
        P = self.p_model.predict(tree_emb)
        P = self.encoded_labels.inverse_transform(P)

        return P

    def transform(self, X):

        tree_emb = np.hstack([R.proximity(X) for R in self.Rs])

        return tree_emb

    def emb_transform(self, X):

        tree_emb = np.hstack([R.proximity(X) for R in self.Rs])
        tree_emb = self.tree_emb.transform(tree_emb)
        tree_emb = self.R_PCA.transform(tree_emb)

        return tree_emb

    def approx_emb(self, X):
        
        return self.l_model.predict(X)