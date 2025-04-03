import numpy as np
 
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.decomposition import PCA

from scipy.sparse import csr_array
from scipy.sparse import hstack as sp_hstack

from umap import UMAP

from LANDMarkClassifier import LANDMarkClassifier

from matplotlib import pyplot as plt

import seaborn as sns

from .transformers_treeord import CLRClosureTransformer, ResampleRandomizeTransform
from .feature_importance_treeord import ExplainImportance


def basic_transform(X, transformer, exclude_col):

    if X.ndim == 2:
        X_in = np.copy(X, "C")

    else:
        X_in = np.asarray([X])

    # Transform data
    if exclude_col[0]:
        excl_range = np.asarray(exclude_col[1])

        X_transformed = np.delete(X_in, excl_range, axis=1)

        if isinstance(transformer, type(None)) == False:
            X_transformed = transformer.transform(X_transformed)

        X_no_transform = X_in[:, excl_range]

        if X_transformed.ndim == 1:
            X_transformed = X_transformed.reshape(-1,1)

        if X_no_transform.ndim == 1:
             X_no_transform = X_no_transform.rehsape(-1,1)

        X_transformed = np.hstack((X_transformed, X_no_transform))

    else:
        X_transformed = X_in

        if isinstance(transformer, type(None)) == False:
            X_transformed = transformer.transform(X_in)

    if X.ndim == 2:
        return X_transformed

    else:
        return X_transformed[0]


# Tree Ordination class
class TreeOrdination(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        feature_names,
        resampler=None,
        metric="hamming",
        prox_method = "terminal",
        supervised_clf=ExtraTreesClassifier(1024),
        proxy_model = ExtraTreesRegressor(1024),
        landmark_model = LANDMarkClassifier(160, use_nnet=False, n_jobs = 8),
        n_iter_unsup=5,
        transformer=None,
        exclude_col=[False, 0],
        n_neighbors=8,
        n_components=2,
        min_dist=0.001,
    ):
        self.feature_names = feature_names

        self.resampler = resampler

        self.metric = metric
        self.prox_method = prox_method

        self.supervised_clf = supervised_clf
        self.proxy_model = proxy_model
        self.landmark_model = landmark_model

        self.n_iter_unsup = n_iter_unsup

        self.transformer = transformer

        self.exclude_col = exclude_col

        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist

    def get_initial_embedding(self, X):

        # Get an Initial LANDMark Representations
        self.Rs = []
        self.LM_emb = []
        self.transformers = []

        for i in range(self.n_iter_unsup):

            print("Iteration %d..." %i)

            #No resample and transform
            resampler = ResampleRandomizeTransform(self.resampler,
                                                   self.transformer,
                                                   self.exclude_col)

            X_rand, y_rand = resampler.fit_resample(X, self.y)

            # Train model
            model = clone(self.landmark_model)

            model.fit(X_rand, y_rand)
            self.Rs.append(model)

            # Get proximity
            X_trf = resampler.transform(X)

            # Update Overall Proximity
            if i > 0:
                self.LM_emb = np.hstack((self.LM_emb, model.proximity(X_trf, self.prox_method).toarray())).astype(np.int8) #sp_hstack((self.LM_emb, model.proximity(X_trf, self.prox_method)))
            else:
                self.LM_emb = model.proximity(X_trf, self.prox_method).toarray().astype(np.int8)

            # Save the resampler
            self.transformers.append(resampler)

        # Get Embeddings
        self.UMAP_trf = UMAP(
            n_neighbors=self.n_neighbors,
            n_components=15,
            min_dist=self.min_dist,
            metric="hamming",
            densmap=False,
        ).fit(self.LM_emb)

        self.UMAP_emb = self.UMAP_trf.transform(self.LM_emb)

        self.PCA_trf = PCA(self.n_components, whiten = True).fit(self.UMAP_emb)

        self.PCA_emb = self.PCA_trf.transform(self.UMAP_emb)

    def fit(self, X, y=None):

        # Save X and y
        self.X = np.copy(X, "C")
        self.y = np.copy(y, "C")

        # Encode y
        self.encoded_labels = LabelEncoder().fit(self.y)
        y_enc = self.encoded_labels.transform(self.y)

        # Get initial embeddings, features, and cluster
        self.f_name = np.asarray(self.feature_names)

        self.get_initial_embedding(X)

        # Train a Classification model
        self.p_model = clone(self.supervised_clf).fit(self.LM_emb, y_enc.astype(int))

        # Train a projection model (Approximate Transformer)
        if self.exclude_col[0]:
            excl_range = np.asarray(self.exclude_col[1])

            X_transformed = np.delete(X, excl_range, axis=1)

            if isinstance(self.transformer, type(None)) == False:
                self.proj_transformer = clone(self.transformer)

                X_transformed = self.proj_transformer.fit_transform(X_transformed)

            else:
                self.proj_transformer = None

            X_no_transform = X[:, excl_range]

            X_transformed = np.hstack((X_transformed, X_no_transform))

        else:
            X_transformed = np.copy(X, "C")

            if isinstance(self.transformer, type(None)) == False:
                self.proj_transformer = clone(self.transformer)

                X_transformed = self.proj_transformer.fit_transform(X)

            else:
                self.proj_transformer = None

        self.l_model = clone(self.proxy_model).fit(
            X_transformed, self.PCA_emb
        )

        return self

    def get_importance(self):

        self.feature_importance_explainer = ExplainImportance(self.l_model, self.f_name)

        self.feature_importance_explainer.get_importance()

    def plot_importance_global(
        self, X, y, class_name, n_features=10, axis=0, summary="median", **kwargs
    ):

        # Transform Samples
        X_transformed = basic_transform(X, self.proj_transformer, self.exclude_col)

        # Grab samples for a specific class, if supplied
        y_locs = np.where(y == class_name, True, False)

        return self.feature_importance_explainer.plot_importance(
            X_transformed[y_locs], n_features, class_name, axis, summary, **kwargs
        )

    def plot_importance_local(self, X, n_features=10, axis=0, **kwargs):

        # Transform data
        X_transformed = basic_transform(X, self.proj_transformer, self.exclude_col)

        return self.feature_importance_explainer.plot_importance(
            X_transformed,
            n_features,
            class_name=None,
            axis=axis,
            summary=None,
            **kwargs
        )

    def plot_projection(self, X, y, ax_1=0, ax_2=1, use_approx=True, trf_type = "PCA"):

        # Plot data
        projection = self.emb_transform(X, trf_type)

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.scatterplot(x=projection[:, ax_1], y=projection[:, ax_2], hue=y, ax=ax)

        pc_ax_1 = self.PCA_trf.explained_variance_ratio_[ax_1] * 100
        pc_ax_2 = self.PCA_trf.explained_variance_ratio_[ax_2] * 100

        ax.set_xlabel("PCA %s (%.3f Percent)" % (str(ax_1 + 1), pc_ax_1))
        ax.set_ylabel("PCA %s (%.3f Percent)" % (str(ax_2 + 1), pc_ax_2))

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        return fig, ax

    def predict_proba(self, X):

        tree_emb = self.emb_transform(X, "UMAP")

        P = self.p_model.predict_proba(tree_emb)

        return P

    def predict(self, X):

        tree_emb = self.emb_transform(X, "LM")

        P = self.p_model.predict(tree_emb)

        P = self.encoded_labels.inverse_transform(P)

        return P

    def emb_transform(self, X, trf_type = "PCA"):

        if trf_type == "approx":
            X_transformed = basic_transform(X, self.proj_transformer, self.exclude_col)

            return self.l_model.predict(X_transformed)

        else:
            tree_emb = []

            for i in range(self.n_iter_unsup):

                # Get trained model
                model = self.Rs[i]
                transformer = self.transformers[i]

                # Get proximity
                if i != 0:
                    tree_emb = np.hstack((tree_emb, model.proximity(transformer.transform(X), self.prox_method).toarray()))#sp_hstack((tree_emb, model.proximity(transformer.transform(X), self.prox_method)))
                else:
                    tree_emb = model.proximity(transformer.transform(X), self.prox_method).toarray()

            if trf_type == "LM":
                return tree_emb

            umap_emb = self.UMAP_trf.transform(tree_emb)
            if trf_type == "UMAP":
                return umap_emb

            pca_emb = self.PCA_trf.transform(umap_emb)
            if trf_type == "PCA":
                return pca_emb
