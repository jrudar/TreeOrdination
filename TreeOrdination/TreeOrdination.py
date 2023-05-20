import numpy as np
 
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.decomposition import PCA

from umap import UMAP

from LANDMark import LANDMarkClassifier

from matplotlib import pyplot as plt

import seaborn as sns

from .transformers_treeord import NoTransform, CLRClosureTransformer, NoResample, ResampleRandomizeTransform
from .feature_importance_treeord import ExplainImportance


def basic_transform(X, transformer, exclude_col):

    if X.ndim == 2:
        X_in = X

    else:
        X_in = np.asarray([X])

    # Transform data
    if exclude_col[0]:
        excl_range = np.asarray(exclude_col[1])

        X_transformed = np.delete(X_in, excl_range, axis=1)

        X_transformed = transformer.transform(X_transformed)

        X_no_transform = X_in[:, excl_range]

        X_transformed = np.hstack((X_transformed, X_no_transform))

    else:
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
        resampler=NoResample(),
        metric="hamming",
        supervised_clf=ExtraTreesClassifier(1024),
        proxy_model = ExtraTreesRegressor(1024),
        n_iter_unsup=5,
        unsup_n_estim=160,
        transformer=NoTransform(),
        exclude_col=[False, 0],
        n_neighbors=8,
        n_components=2,
        min_dist=0.001,
        n_jobs=4,
    ):
        self.feature_names = feature_names

        self.resampler = resampler

        self.metric = metric

        self.supervised_clf = supervised_clf
        self.proxy_model = proxy_model

        self.n_iter_unsup = n_iter_unsup
        self.unsup_n_estim = unsup_n_estim

        self.transformer = transformer

        self.exclude_col = exclude_col

        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist

        self.n_jobs = n_jobs

    def get_initial_embedding(self, X):

        # Get an Initial LANDMark Representations
        self.Rs = []
        self.R_final = []

        for i in range(self.n_iter_unsup):

            # Prepare resampling, randomization, and transformation object
            resampler = ResampleRandomizeTransform(clone(self.resampler),
                                                   clone(self.transformer),
                                                   self.exclude_col)

            # Train model
            model = LANDMarkClassifier(
                self.unsup_n_estim,
                use_nnet=False,
                n_jobs=self.n_jobs,
                resampler = resampler
            ).fit(X, y) # The resampler is created in such a way that it overrides the default behavior of the LANDMarkClassifier. y is only used to ensure the distribuion of samples in the re-sampled data is matches that specified by the user
            self.Rs.append(model)

            # Get proximity
            self.R_final.append(model.proximity(X))

        # Get Overall Proximity
        self.R_final = np.hstack(self.R_final)

        # Get Embeddings
        self.tree_emb = UMAP(
            n_neighbors=self.n_neighbors,
            n_components=15,
            min_dist=self.min_dist,
            metric=self.metric,
            densmap=False,
        ).fit(self.R_final)

        self.R_PCA = PCA(self.n_components).fit(self.tree_emb.transform(self.R_final))

        self.R_PCA_emb = self.R_PCA.transform(self.tree_emb.transform(self.R_final))

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
        self.p_model = clone(self.supervised_clf).fit(self.R_final, y_enc.astype(int))

        # Train a projection model (Approximate Transformer)
        if self.exclude_col[0]:
            excl_range = np.asarray(self.exclude_col[1])

            X_transformed = np.delete(X, excl_range, axis=1)

            self.proj_transformer = clone(self.transformer)

            X_transformed = self.proj_transformer.fit_transform(X_transformed)

            X_no_transform = X[:, excl_range]

            X_transformed = np.hstack((X_transformed, X_no_transform))

        else:
            self.proj_transformer = clone(self.transformer)

            X_transformed = self.proj_transformer.fit_transform(X)

        self.l_model = clone(self.proxy_model).fit(
            X_transformed, self.R_PCA_emb
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

    def plot_projection(self, X, y, ax_1=0, ax_2=1, use_approx=True):

        # Plot data
        if use_approx is False:
            projection = self.emb_transform(X)

        else:
            projection = self.approx_emb(X)

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.scatterplot(x=projection[:, ax_1], y=projection[:, ax_2], hue=y, ax=ax)

        pc_ax_1 = self.R_PCA.explained_variance_ratio_[ax_1] * 100
        pc_ax_2 = self.R_PCA.explained_variance_ratio_[ax_2] * 100

        ax.set_xlabel("PCA %s (%.3f Percent)" % (str(ax_1 + 1), pc_ax_1))
        ax.set_ylabel("PCA %s (%.3f Percent)" % (str(ax_2 + 1), pc_ax_2))

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        return fig, ax

    def predict_proba(self, X):

        tree_emb = self.transform(X)

        P = self.p_model.predict_proba(tree_emb)

        return P

    def predict(self, X):

        tree_emb = self.transform(X)

        P = self.p_model.predict(tree_emb)

        P = self.encoded_labels.inverse_transform(P)

        return P

    def transform(self, X):

        tree_emb = []

        for i in range(self.n_iter_unsup):

            # Get trained model
            model = self.Rs[i]

            # Get proximity
            proximity = model.proximity(X)

            tree_emb.append(proximity)

        tree_emb = np.hstack(tree_emb)

        return tree_emb

    def emb_transform(self, X):

        tree_emb = self.transform(X)

        tree_emb = self.tree_emb.transform(tree_emb)

        tree_emb = self.R_PCA.transform(tree_emb)

        return tree_emb

    def approx_emb(self, X):

        # Transform data and return prediction
        X_transformed = basic_transform(X, self.proj_transformer, self.exclude_col)

        return self.l_model.predict(X_transformed)
