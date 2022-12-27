import numpy as np

import pandas as pd

# For class construction
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import resample
from sklearn.decomposition import PCA

from skbio.stats.composition import multiplicative_replacement, closure, clr

from umap import UMAP

from LANDMark import LANDMarkClassifier

from deicode.preprocessing import rclr
from deicode.matrix_completion import MatrixCompletion

#Randomization function
def addcl2(X, scale, clr_trf, rclr_trf):

    #Resample rows
    X_resamp = resample(X, replace = True, n_samples = X.shape[0])

    #Resampling can introduce features with zeros. Remove
    zeros = np.sum(X_resamp, axis = 0)
    zeros = np.where(zeros > 0, True, False)
    X_resamp = X_resamp[:, zeros]
    
    #Resample columns
    X_perm = np.copy(X_resamp, "C")
    for col in range(X_perm.shape[0]):
        X_perm[:, col] = np.random.choice(X_perm[:, col], replace = False, size = X_perm.shape[0])
        
    #Create Labels
    y_new = [0 for _ in range(X_resamp.shape[0])]
    y_new.extend([1 for _ in range(X_resamp.shape[0])])
    y_new = np.asarray(y_new)
    
    #Create merged dataset
    X_new = np.vstack((X_resamp, X_perm))
            
    #Scale so sum of rows is unity
    if scale:
        X_new = closure(X_new)

    #Apply CLR ratio transformation
    if clr_trf:
        if scale:
            X_new = clr(multiplicative_replacement(X_new))

        else:
            X_new = clr(multiplicative_replacement(closure(X_new)))

    #Apply RCLR transformation
    if rclr_trf:
        X_prop_train = rclr(X_new.transpose()).transpose()
        M = MatrixCompletion(2, max_iterations = 1000).fit(X_new)
        X_new = M.solution

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
        max_samples_tree = 75,
        n_jobs = 4,

        scale = False,
        clr_trf = False,
        rclr_trf = False,

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
        self.max_samples_tree = max_samples_tree
        self.n_jobs = n_jobs

        self.scale = scale
        self.clr_trf = clr_trf
        self.rclr_trf = rclr_trf

        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        
    def get_initial_embedding(self, X):

        #Get an Initial LANDMark Representation
        self.Rs = [LANDMarkClassifier(self.unsup_n_estim, use_nnet = False, max_samples_tree = self.max_samples_tree, n_jobs = self.n_jobs).fit(*addcl2(X, self.scale, self.clr_trf, self.rclr_trf)) for _ in range(self.n_iter_unsup)]

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
        if self.scale == True or self.clr_trf == True or self.rclr_trf == True:
            X_new = self.scale_clr(X)

        else:
            X_new = X

        self.l_model = clone(ExtraTreesRegressor(1024)).fit(X_new, self.R_PCA_emb)

        return self

    def scale_clr(self, X):

        X_new = np.copy(X, order = "C")

        #Scale so sum of rows is unity
        if self.scale:
            X_new = closure(X_new)

        #Apply CLR ratio transformation
        if self.clr_trf:
            if self.scale:
                X_new = clr(multiplicative_replacement(X_new))

            else:
                X_new = clr(multiplicative_replacement(closure(X_new)))

        #Apply RCLR transformation
        if self.rclr_trf:
            X_new = rclr(X_new.transpose()).transpose()
            M = MatrixCompletion(2, max_iterations = 1000).fit(X_new)
            X_new = M.solution

        return X_new

    def predict_proba(self, X):

        if self.scale == True or self.clr_trf == True or self.rclr_trf == True:
            X_new = self.scale_clr(X)

        else:
            X_new = X

        tree_emb = np.hstack([R.proximity(X_new) for R in self.Rs])
        P = self.p_model.predict_proba(tree_emb)

        return P

    def predict(self, X):

        if self.scale == True or self.clr_trf == True or self.rclr_trf == True:
            X_new = self.scale_clr(X)

        else:
            X_new = X

        tree_emb = np.hstack([R.proximity(X_new) for R in self.Rs])
        P = self.p_model.predict(tree_emb)
        P = self.encoded_labels.inverse_transform(P)

        return P

    def transform(self, X):

        if self.scale == True or self.clr_trf == True or self.rclr_trf == True:
            X_new = self.scale_clr(X)

        else:
            X_new = X

        tree_emb = np.hstack([R.proximity(X_new) for R in self.Rs])

        return tree_emb

    def emb_transform(self, X):

        if self.scale == True or self.clr_trf == True or self.rclr_trf == True:
            X_new = self.scale_clr(X)

        else:
            X_new = X

        tree_emb = np.hstack([R.proximity(X_new) for R in self.Rs])
        tree_emb = self.tree_emb.transform(tree_emb)
        tree_emb = self.R_PCA.transform(tree_emb)

        return tree_emb

    def approx_emb(self, X):
        
        if self.scale == True or self.clr_trf == True or self.rclr_trf == True:
            X_new = self.scale_clr(X)

        else:
            X_new = X

        return self.l_model.predict(X_new)
