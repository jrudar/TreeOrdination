from shap import Explainer

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd


class ExplainImportance:
    """
    Returns an Explanation Model.
    """

    def __init__(self, model, feature_names):

        self.model = model
        self.feature_names = feature_names

    def get_importance(self):

        # TO-DO: Add KernelExplainer for models not based on trees.
        #        Add explainer for neural networks

        self.explainer = Explainer(self.model)

    def plot_importance(
        self, X, n_features, class_name, axis=0, summary="median", **kwargs
    ):
        """
        Create a horizontal barchart of feature effects, sorted by their magnitude.

        Adapted from: https://docs.seldon.io/projects/alibi/en/stable/examples/path_dependent_tree_shap_adult_xgb.html
        """
        # Calculate importances (samples, features, outputs or samples, features)
        feat_imp = self.explainer.shap_values(X)#[:, :, axis]

        # Determine if Global or Local feature importances are to be printed
        # Determine how SHAP values are to be sumarized across multiple samples from each class
        if feat_imp.ndim > 2:
            overview = "Global"

            ylabel = "Feature Name"

            if summary == "median":
                feat_imp = np.median(feat_imp, axis=0)[:, axis]
                xlabel = "Median(SHAP Values)"

            elif summary == "mean":
                feat_imp = np.mean(feat_imp, axis=0)[:, axis]
                xlabel = "Mean(SHAP Values)"

            elif summary == "abs_median":
                feat_imp = np.median(np.abs(feat_imp), axis=0)[:, axis]
                xlabel = "Median(|SHAP Values|)"

            elif summary == "abs_mean":
                feat_imp = np.mean(np.abs(feat_imp), axis=0)[:, axis]
                xlabel = "Mean(|SHAP Values|)"

        else:
            overview = "Local"
            xlabel = "SHAP Values"
            ylabel = "Feature Name (Magnitude)"
            feat_imp = feat_imp[:, axis]
            prediction = self.model.predict([X])[0][axis]

        # Sort Features
        feat_imp_pd = pd.Series(data=feat_imp, index=self.feature_names)
        feat_imp_sorted_index = (
            feat_imp_pd.abs().sort_values(ascending=False).index.values[0:n_features]
        )
        feat_imp_sorted_data = feat_imp_pd[feat_imp_sorted_index].values

        if X.ndim == 1:
            feat_values = pd.Series(data=X, index=self.feature_names)[
                feat_imp_sorted_index
            ].values

            feat_imp_sorted_index = np.asarray(
                [
                    "%s (%.2f)" % (f_name, feat_values[i])
                    for i, f_name in enumerate(feat_imp_sorted_index)
                ]
            )

        # Get colors of bars
        bar_cols = [
            "Orange" if score < 0 else "Green" for score in feat_imp_sorted_data
        ]

        # Some plot aesthetics
        labels_fontsize = kwargs.get("labels_fontsize", 10)
        tick_labels_fontsize = kwargs.get("tick_labels_fontsize", 10)

        # plot
        fig, ax = plt.subplots(figsize=(10, 5))
        y_pos = np.arange(len(feat_imp_sorted_data))
        ax.barh(y_pos, feat_imp_sorted_data, color=bar_cols)

        # set lables
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_imp_sorted_index, fontsize=tick_labels_fontsize)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=labels_fontsize)
        ax.set_ylabel(ylabel, fontsize=labels_fontsize)

        if overview == "Global":
            plt.title(
                "TreeOrdination %s Feature Importance Plot for Class %s"
                % (overview, str(class_name))
            )

        else:
            plt.title(
                "TreeOrdination %s Feature Importance Plot\nPredicted Model Output for Sample = %.2f"
                % (overview, prediction)
            )

        return ax, fig
