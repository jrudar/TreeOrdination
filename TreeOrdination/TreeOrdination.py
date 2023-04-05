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

def scaler(X, scale, clr_trf, rclr_trf, n_comp = 2):

    #Scale so sum of rows is unity
    if scale:
        X_new = closure(X)

        return X_new

    #Apply CLR ratio transformation
    if clr_trf:
        if scale:
            X_new = clr(multiplicative_replacement(X))

            return X_new

        else:
            X_new = clr(multiplicative_replacement(closure(X)))

            return X_new

    #Apply RCLR transformation
    if rclr_trf:
        X_prop_train = rclr(X_new.transpose()).transpose()
        M = MatrixCompletion(n_comp, max_iterations = 1000).fit(X)
        X_new = M.solution

        return X_new

    return X

#Randomization function
def addcl2(X, scale, clr_trf, rclr_trf, exclude_col):

    #Resample rows
    X_resamp = resample(X, replace = True, n_samples = X.shape[0])

    #If there are columns to exclude from scaling...
    if exclude_col[0]:

        #Exclude columns immediatly
        excl_range = np.asarray(exclude_col[1])

        X_resamp_scale = np.delete(X_resamp, excl_range, axis = 1)
        X_resamp_noscale = X_resamp[:, excl_range]

        #Resampling can introduce features with zeros. Remove
        zeros_scale = np.sum(X_resamp_scale, axis = 0)
        zeros_scale = np.where(zeros_scale > 0, True, False)
        X_resamp_scale = X_resamp_scale[:, zeros_scale]
    
        #Resample columns
        X_perm_scale = np.copy(X_resamp_scale, "C")
        for col in range(X_perm_scale.shape[1]):
            X_perm_scale[:, col] = np.random.choice(X_perm_scale[:, col], replace = False, size = X_perm_scale.shape[0])

        #Scale data
        X_resamp_scale = scaler(np.vstack((X_resamp_scale, X_perm_scale)), scale, clr_trf, rclr_trf)

        #Resampling can introduce features with zeros. Remove
        zeros_noscale = np.sum(X_resamp_noscale, axis = 0)
        zeros_noscale = np.where(zeros_noscale > 0, True, False)
        X_resamp_noscale = X_resamp_noscale[:, zeros_noscale]

        #Resample columns
        X_perm_noscale = np.copy(X_resamp_noscale, "C")
        for col in range(X_perm_noscale.shape[1]):
            X_perm_noscale[:, col] = np.random.choice(X_perm_noscale[:, col], replace = False, size = X_perm_noscale.shape[0])

        X_resamp_noscale = np.vstack((X_resamp_noscale, X_perm_noscale))

        X_new = np.hstack((X_resamp_scale, X_resamp_noscale))

        #Create Labels
        y_new = [0 for _ in range(X_resamp.shape[0])]
        y_new.extend([1 for _ in range(X_resamp.shape[0])])
        y_new = np.asarray(y_new)

        return X_new, y_new, zeros_scale, zeros_noscale

    else:
        #Resampling can introduce features with zeros. Remove
        zeros_scale = np.sum(X_resamp_scale, axis = 0)
        zeros_scale = np.where(zeros_scale > 0, True, False)
        X_resamp_scale = X_resamp_scale[:, zeros_scale]

        #Resample columns
        X_perm = np.copy(X_resamp_scale, "C")
        for col in range(X_perm.shape[1]):
            X_perm[:, col] = np.random.choice(X_perm[:, col], replace = False, size = X_perm.shape[0])

        X_new = np.vstack((X_resamp_scale, X_perm))

        #Create Labels
        y_new = [0 for _ in range(X_resamp_scale.shape[0])]
        y_new.extend([1 for _ in range(X_resamp_scale.shape[0])])
        y_new = np.asarray(y_new)

        return X_new, y_new, zeros_scale, np.asarray([])

#Tree Ordination class
class TreeOrdination(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        
        feature_names,

        resample_data = False,
        resample_class = None,
        n_resamples = None,

        metric = "hamming",

        supervised_clf = ExtraTreesClassifier(1024),
        n_iter_unsup = 5,
        unsup_n_estim = 160,
        max_samples_tree = 100,
        n_jobs = 4,

        scale = False,
        clr_trf = False,
        rclr_trf = False,
        exclude_col = [False, 0],

        n_neighbors = 8,
        n_components = 2,
        min_dist = 0.001,
        
    ):
        self.feature_names = feature_names

        self.resample_data = resample_data
        self.resample_class = resample_class
        self.n_resamples = n_resamples

        self.metric = metric

        self.supervised_clf = supervised_clf
        self.n_iter_unsup = n_iter_unsup
        self.unsup_n_estim = unsup_n_estim
        self.max_samples_tree = max_samples_tree
        self.n_jobs = n_jobs

        self.scale = scale
        self.clr_trf = clr_trf
        self.rclr_trf = rclr_trf
        self.exclude_col = exclude_col

        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        
    def get_initial_embedding(self, X):

        #Get an Initial LANDMark Representations
        self.Rs = []
        self.R_final = []
        self.features_scaled = []
        self.features_unscaled = []

        for i in range(self.n_iter_unsup):

            #Resample
            if self.resample_data:
                re_loc = np.where(self.y == self.resample_class, True, False)
                re_not_loc = np.where(self.y != self.resample_class, True, False)

                X_re = X[re_loc]
                X_not_re = X[re_not_loc]

                X_re = resample(X_re, replace = False, n_samples = self.n_resamples, random_state = i)

                X_prep = np.vstack((X_re, X_not_re))

            else:
                X_prep = X

            #Get random features
            X_rnd, y_rnd, scaled_features, unscaled_features = addcl2(X_prep, self.scale, self.clr_trf, self.rclr_trf, self.exclude_col)

            #Save non-zero features for later use
            self.features_scaled.append(scaled_features)
            self.features_unscaled.append(unscaled_features)

            #Train model
            model = LANDMarkClassifier(self.unsup_n_estim, use_nnet = False, max_samples_tree = self.max_samples_tree, n_jobs = self.n_jobs).fit(X_rnd, y_rnd)
            self.Rs.append(model)

            #Get proximity
            if self.exclude_col[0]:
                excl_range = np.asarray(self.exclude_col[1])

                X_scale = np.delete(X, excl_range, axis = 1)[:, scaled_features]
                X_scale = scaler(X_scale, self.scale, self.clr_trf, self.rclr_trf)

                X_noscale = X[:, excl_range][:, unscaled_features]

                X_reduced = np.hstack((X_scale, X_noscale))

            else:
                X_reduced = X[:, scaled_features]
                X_reduced = scaler(X_reduced, self.scale, self.clr_trf, self.rclr_trf)

            proximity = model.proximity(X_reduced)
            self.R_final.append(proximity)

        #Get Overall Proximity
        self.R_final = np.hstack(self.R_final)
        
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
        if self.exclude_col[0]:
            excl_range = np.asarray(self.exclude_col[1])

            X_scale = np.delete(X, excl_range, axis = 1)
            X_scale = scaler(X_scale, self.scale, self.clr_trf, self.rclr_trf)

            X_noscale = X[:, excl_range]

            X_reduced = np.hstack((X_scale, X_noscale))

        else:
            X_reduced = scaler(X, self.scale, self.clr_trf, self.rclr_trf)

        self.l_model = clone(ExtraTreesRegressor(1024)).fit(X_reduced, self.R_PCA_emb)

        return self

    def predict_proba(self, X):

        tree_emb = []

        for i in range(self.n_iter_unsup):

            #Grab locations of non-zero features
            scaled_features = self.features_scaled[i]
            unscaled_features = self.features_unscaled[i]

            #Train model
            model = self.Rs[i]

            #Get proximity
            if self.exclude_col[0]:
                excl_range = np.asarray(self.exclude_col[1])

                X_scale = np.delete(X, excl_range, axis = 1)[:, scaled_features]

                if X_scale.ndim == 1:
                    X_scale = np.asarray([X_scale])

                X_scale = scaler(X_scale, self.scale, self.clr_trf, self.rclr_trf)
                
                if X_scale.ndim == 1:
                    X_scale = np.asarray([X_scale])

                X_noscale = X[:, excl_range][:, unscaled_features]

                X_reduced = np.hstack((X_scale, X_noscale))

            else:
                X_reduced = X[:, scaled_features]
                X_reduced = scaler(X_reduced, self.scale, self.clr_trf, self.rclr_trf)

            proximity = model.proximity(X_reduced)

            tree_emb.append(proximity)

        tree_emb = np.hstack(tree_emb)

        P = self.p_model.predict_proba(tree_emb)

        return P

    def predict(self, X):

        tree_emb = []

        for i in range(self.n_iter_unsup):

            #Grab locations of non-zero features
            scaled_features = self.features_scaled[i]
            unscaled_features = self.features_unscaled[i]

            #Train model
            model = self.Rs[i]

            #Get proximity
            if self.exclude_col[0]:
                excl_range = np.asarray(self.exclude_col[1])

                X_scale = np.delete(X, excl_range, axis = 1)[:, scaled_features]

                if X_scale.ndim == 1:
                    X_scale = np.asarray([X_scale])

                X_scale = scaler(X_scale, self.scale, self.clr_trf, self.rclr_trf)
                
                if X_scale.ndim == 1:
                    X_scale = np.asarray([X_scale])

                X_noscale = X[:, excl_range][:, unscaled_features]

                X_reduced = np.hstack((X_scale, X_noscale))

            else:
                X_reduced = X[:, scaled_features]
                X_reduced = scaler(X_reduced, self.scale, self.clr_trf, self.rclr_trf)

            proximity = model.proximity(X_reduced)

            tree_emb.append(proximity)

        tree_emb = np.hstack(tree_emb)

        P = self.p_model.predict(tree_emb)
        
        P = self.encoded_labels.inverse_transform(P)

        return P

    def transform(self, X):

        tree_emb = []

        for i in range(self.n_iter_unsup):

            #Grab locations of non-zero features
            scaled_features = self.features_scaled[i]
            unscaled_features = self.features_unscaled[i]

            #Train model
            model = self.Rs[i]

            #Get proximity
            if self.exclude_col[0]:
                excl_range = np.asarray(self.exclude_col[1])

                X_scale = np.delete(X, excl_range, axis = 1)[:, scaled_features]

                if X_scale.ndim == 1:
                    X_scale = np.asarray([X_scale])

                X_scale = scaler(X_scale, self.scale, self.clr_trf, self.rclr_trf)
                
                if X_scale.ndim == 1:
                    X_scale = np.asarray([X_scale])

                X_noscale = X[:, excl_range][:, unscaled_features]

                X_reduced = np.hstack((X_scale, X_noscale))

            else:
                X_reduced = X[:, scaled_features]
                X_reduced = scaler(X_reduced, self.scale, self.clr_trf, self.rclr_trf)

            proximity = model.proximity(X_reduced)

            tree_emb.append(proximity)

        tree_emb = np.hstack(tree_emb)

        return tree_emb

    def emb_transform(self, X):

        tree_emb = []

        for i in range(self.n_iter_unsup):

            #Grab locations of non-zero features
            scaled_features = self.features_scaled[i]
            unscaled_features = self.features_unscaled[i]

            #Train model
            model = self.Rs[i]

            #Get proximity
            if self.exclude_col[0]:
                excl_range = np.asarray(self.exclude_col[1])

                X_scale = np.delete(X, excl_range, axis = 1)[:, scaled_features]

                if X_scale.ndim == 1:
                    X_scale = np.asarray([X_scale])

                X_scale = scaler(X_scale, self.scale, self.clr_trf, self.rclr_trf)
                
                if X_scale.ndim == 1:
                    X_scale = np.asarray([X_scale])

                X_noscale = X[:, excl_range][:, unscaled_features]

                X_reduced = np.hstack((X_scale, X_noscale))

            else:
                X_reduced = X[:, scaled_features]
                X_reduced = scaler(X_reduced, self.scale, self.clr_trf, self.rclr_trf)

            proximity = model.proximity(X_reduced)

            tree_emb.append(proximity)

        tree_emb = np.hstack(tree_emb)

        tree_emb = self.tree_emb.transform(tree_emb)
        
        tree_emb = self.R_PCA.transform(tree_emb)

        return tree_emb

    def approx_emb(self, X):
        
        #Train a projection model
        if self.exclude_col[0]:
            excl_range = np.asarray(self.exclude_col[1])

            X_scale = np.asarray([np.delete(X, excl_range, axis = 1)])
            X_scale = np.asarray([scaler(X_scale, self.scale, self.clr_trf, self.rclr_trf)])

            X_noscale = X[:, excl_range]

            X_reduced = np.hstack((X_scale, X_noscale))

        else:
            X_reduced = scaler(X, self.scale, self.clr_trf, self.rclr_trf)

        return self.l_model.predict(X_reduced)