#!/usr/bin/env python
# coding: utf-8

from __future__ import division

# utils
import pandas as pd
import numpy as np
from collections import Counter

# blocks
from scipy.stats import norm
from numpy.random import poisson, lognormal
from skbio.stats.composition import closure
from scipy.special import kl_div
from scipy.stats import entropy

# minimize model perams
from sklearn.metrics import mean_squared_error, balanced_accuracy_score
from scipy.optimize import minimize

# Import relevant libraries
from skbio.stats.distance import permanova, DistanceMatrix
from skbio.stats.ordination import pcoa
from skbio.stats import subsample_counts
from skbio.stats.composition import multiplicative_replacement, closure, clr

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import pairwise_distances, balanced_accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.ensemble import ExtraTreesClassifier

import seaborn as sns

import matplotlib.pyplot as plt

# For printing graphs
from itertools import combinations
from statannotations.Annotator import Annotator

from LANDMark import LANDMarkClassifier
from TreeOrdination import TreeOrdination

# For RCLR and RPCA
from deicode.preprocessing import rclr
from deicode.matrix_completion import MatrixCompletion

from numpy.random import RandomState

from umap import UMAP

# Function for rarefaction
# https://stackoverflow.com/questions/15507993/quickly-rarefy-a-matrix-in-numpy-python
def rarefaction(M, y1, M2, y2, D, seed=0):
    prng = RandomState(seed)  # reproducible results

    n_occur = M.sum(axis=1)
    rem = np.where(n_occur < depth, False, True)
    M_ss = M[rem]
    n_occur = n_occur[rem]
    nvar = M.shape[1]  # number of variables

    # Do training data
    Mrarefied = np.empty_like(M_ss)
    for i in range(M_ss.shape[0]):  # for each sample
        p = M_ss[i] / float(n_occur[i])  # relative frequency / probability
        choice = prng.choice(nvar, D, p=p)
        Mrarefied[i] = np.bincount(choice, minlength=nvar)

    # Do testing data
    n_occur2 = M2.sum(axis=1)
    rem2 = np.where(n_occur2 < depth, False, True)
    M_ss2 = M2[rem2]
    n_occur2 = n_occur2[rem2]
    nvar = M2.shape[1]  # number of variables

    Mrarefied2 = np.empty_like(M_ss2)
    for i in range(M_ss2.shape[0]):  # for each sample
        p = M_ss2[i] / float(n_occur2[i])  # relative frequency / probability
        choice = prng.choice(nvar, D, p=p)
        Mrarefied2[i] = np.bincount(choice, minlength=nvar)

    return Mrarefied, y1[rem], Mrarefied2, y2[rem2]


# Function for creating random data for use in unsupervised learning
def addcl2(X, y):

    X_perm = np.copy(X, "C")
    for col in range(X_perm.shape[0]):
        X_perm[:, col] = np.random.choice(
            X_perm[:, col], replace=False, size=X_perm.shape[0]
        )

    y_new = ["Original" for _ in range(X.shape[0])]
    y_new.extend(["Randomized" for _ in range(X.shape[0])])
    y_new = np.asarray(y_new)

    X_new = np.vstack((X, X_perm))

    return X_new, y_new


def get_result_model(model, X_tr, y_tr, X_te, y_te):
    # Fit the model
    model = model.fit(X_tr, y_tr)

    # Predict class labels
    pred_tests = model.predict(X_te)

    # Return BACC
    ba_tests = balanced_accuracy_score(y_te, pred_tests)

    return ba_tests


def get_result_permanova(X, y, n_rep=999):
    pmanova = permanova(X, y, permutations=n_rep)

    pseudo_f, pval = pmanova.values[4:6]
    R2 = 1 - 1 / (
        1
        + pmanova.values[4]
        * pmanova.values[4]
        / (pmanova.values[2] - pmanova.values[3] - 1)
    )

    return pseudo_f, pval, R2


def get_perm(X, y, metric, transform_type, comp_type, pcoa_trf, n_neighbors=15):

    if pcoa_trf == 0:
        D = DistanceMatrix(pairwise_distances(X, metric=metric).astype(np.float32))

    elif pcoa_trf == 1:
        D = pcoa(
            DistanceMatrix(pairwise_distances(X, metric=metric).astype(np.float32)),
            number_of_dimensions=2,
        ).samples.values

        D = DistanceMatrix(pairwise_distances(D, metric="euclidean").astype(np.float32))

    elif pcoa_trf == 2:
        D = UMAP(
            n_components=2, min_dist=0.001, metric=metric, n_neighbors=n_neighbors
        ).fit_transform(X)

        D = DistanceMatrix(pairwise_distances(D, metric="euclidean").astype(np.float32))

    elif pcoa_trf == 3:
        D = DistanceMatrix(X)

    per_result = get_result_permanova(D, y)

    return (
        transform_type,
        comp_type,
        metric,
        per_result[0],
        per_result[1],
        per_result[2],
    )


def get_classifer(
    X,
    y,
    X_te,
    y_te,
    metric,
    model,
    transform_type,
    comp_type,
    model_type,
    pcoa_trf,
    n_neighbors=8,
):

    if pcoa_trf == 0:
        result = get_result_model(model, X, y, X_te, y_te)

    elif pcoa_trf == 1:
        X_trf = UMAP(
            n_components=2, min_dist=0.001, metric=metric, n_neighbors=n_neighbors
        ).fit(X)
        X_tr = X_trf.transform(X)
        X_test_proj = X_trf.transform(X_te)

        result = get_result_model(model, X_tr, y, X_test_proj, y_te)

   # elif pcoa_trf == 2:
    #    X_tr = closure(X)
     #   X_test_proj = closure(X_te)

      #  result = get_result_model(model, X_tr, y, X_test_proj, y_te)

    #elif pcoa_trf == 3:
     #   X_tr = clr(multiplicative_replacement(closure(X)))
      #  X_test_proj = clr(multiplicative_replacement(closure(X_te)))

       # result = get_result_model(model, X_tr, y, X_test_proj, y_te)

    return transform_type, comp_type, model_type, metric, result


# Create positive and negative controls
if __name__ == "__main__":

    # Read in taxa data
    taxa_tab = pd.read_csv(
        "Diseased Gut/rdp.out.tmp", delimiter="\t", header=None
    ).values

    # Keep all ASVs assigned to Bacteria and Archaea, remove Cyanobacteria and Chloroplasts
    idx = np.where(
        ((taxa_tab[:, 2] == "Bacteria") | (taxa_tab[:, 2] == "Archaea")), True, False
    )
    taxa_tab = taxa_tab[idx]
    idx = np.where(taxa_tab[:, 5] != "Cyanobacteria/Chloroplast", True, False)
    taxa_tab = taxa_tab[idx]
    X_selected = set([x[0] for x in taxa_tab])
    taxa_tab_ss = {x[0]: x for x in taxa_tab}

    # Read in ASV table
    X = pd.read_csv("Diseased Gut/ESV.table", index_col=0, sep="\t")
    X_col = [entry.split("_")[0] for entry in X.columns.values]
    X_features = list(set(X.index.values).intersection(X_selected))
    X_index = [s_name.split("_")[0] for s_name in X.columns.values]
    X_signal = X.transpose()[X_features].values

    # Get names of high confidence features
    n_list = [4, 7, 10, 13, 16, 19]
    X_name = []
    cluster_name = []
    for row in taxa_tab:
        for entry in X_features:
            if row[0] == entry:
                if float(row[n_list[-1]]) > 0.8:
                    X_name.append("%s (%s)" % (row[n_list[-1] - 2], entry))
                    cluster_name.append("%s-%s" % (row[n_list[-1] - 2], entry))
                    break

                elif float(row[n_list[-2]]) > 0.8:
                    X_name.append("%s (%s)" % (row[n_list[-2] - 2], entry))
                    cluster_name.append("%s-%s" % (row[n_list[-2] - 2], entry))
                    break

                elif float(row[n_list[-3]]) > 0.8:
                    X_name.append("%s (%s)" % (row[n_list[-3] - 2], entry))
                    cluster_name.append("%s-%s" % (row[n_list[-3] - 2], entry))
                    break

                elif float(row[n_list[-4]]) > 0.8:
                    X_name.append("%s (%s)" % (row[n_list[-4] - 2], entry))
                    cluster_name.append("%s-%s" % (row[n_list[-4] - 2], entry))
                    break

                elif float(row[n_list[-5]]) > 0.8:
                    X_name.append("%s (%s)" % (row[n_list[-5] - 2], entry))
                    cluster_name.append("%s-%s" % (row[n_list[-5] - 2], entry))
                    break

                else:
                    X_name.append("%s" % entry)
                    cluster_name.append("Unclassified-%s" % entry)
                    break

    # Read in metadata
    meta = pd.read_csv("Diseased Gut/metadata.csv", index_col=0)
    meta = meta[["Sample Name", "Host_disease", "Timepoint"]]

    # Correct locations so they are more informative
    meta["Host_disease"] = np.where(
        meta["Host_disease"] == "CD", "Crohn's Disease", meta["Host_disease"]
    )
    meta["Host_disease"] = np.where(
        meta["Host_disease"] == "RA", "Rheumatoid Arthritis", meta["Host_disease"]
    )
    meta["Host_diseaes"] = np.where(
        meta["Host_disease"] == "MS", "Multiple Sclerosis", meta["Host_disease"]
    )
    meta["Host_disease"] = np.where(
        meta["Host_disease"] == "US", "Ulcerative Colitis", meta["Host_disease"]
    )
    meta["Host_disease"] = np.where(
        meta["Host_disease"] == "HC", "Healthy Control", meta["Host_disease"]
    )

    # Split time points
    y_time_1 = np.where(meta["Timepoint"].astype(int) < 2, True, False)
    y_time_2 = np.where(meta["Timepoint"].astype(int) > 1, True, False)

    meta_1 = meta[y_time_1]
    meta_2 = meta[y_time_2]

    meta = meta_1

    # List of phenotypes/datasets to test
    pheno = "Crohn's Disease-Healthy Control"

    pheno_a, pheno_b = pheno.split("-")

    idx = np.where(
        ((meta["Host_disease"] == pheno_a) | (meta["Host_disease"] == pheno_b)),
        True,
        False,
    )

    meta = meta[idx]
    X_signal = pd.DataFrame(X_signal, index=X_index)
    X_signal = X_signal.loc[meta.index.values]

    # Feature names
    cluster_names = X_features

    experiment = ["Crohns_Disease"]

    # Cross-validation - Loop through signal and random data
    large_test = True
    if large_test:
        for dataset_type, dataset in enumerate([X_signal]):
            # List of balanced accuracy scores (test and validation) and PerMANOVA data for each iteratiion
            BAS_data = []
            PER_data = []

            # 5x5 Stratified Cross-Validation - Positive Control
            splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)

            counter = 1

            for train, tests in splitter.split(dataset, meta["Host_disease"].values):

                print("Iteration Number:", counter)
                counter += 1

                X_train = dataset.values[train]
                X_tests = dataset.values[tests]

                y_train = meta["Host_disease"].values[train]
                y_tests = meta["Host_disease"].values[tests]

                # Retain ASVs present in more than 2 samples
                X_sum = np.where(X_train > 0, 1, 0)
                X_sum = np.sum(X_sum, axis=0)
                removed_ASVs = np.where(X_sum > 2, True, False)

                feature_names = np.asarray(cluster_names)[removed_ASVs]

                X_train = X_train[:, removed_ASVs]
                X_tests = X_tests[:, removed_ASVs]

                # Remove all samples with zero rows
                train_sum = np.where(np.sum(X_train, axis=1) > 0, True, False)
                tests_sum = np.where(np.sum(X_tests, axis=1) > 0, True, False)

                X_train = X_train[train_sum]
                X_tests = X_tests[tests_sum]

                y_train = y_train[train_sum]
                y_tests = y_tests[tests_sum]

                # Rarefy to the 15th percentile
                noccur = np.sum(
                    X_train, axis=1
                )  # number of occurrences for each sample
                depth = int(np.percentile(noccur, float(15.0)))  # sampling depth

                X_train_rare, y_train_rare, X_tests_rare, y_tests_rare = rarefaction(
                    X_train, y_train, X_tests, y_tests, depth, seed=0
                )

                # Get randomized data to build trees - PA, Bray-Curtis
                X_rnd_1, y_rnd_1 = addcl2(X_train_rare, y_train_rare)
                X_rnd_2, y_rnd_2 = addcl2(X_train_rare, y_train_rare)
                X_rnd_3, y_rnd_3 = addcl2(X_train_rare, y_train_rare)
                X_rnd_4, y_rnd_4 = addcl2(X_train_rare, y_train_rare)
                X_rnd_5, y_rnd_5 = addcl2(X_train_rare, y_train_rare)
                X_rnd = np.vstack((X_rnd_1, X_rnd_2, X_rnd_3, X_rnd_4, X_rnd_5))
                y_rnd = np.hstack((y_rnd_1, y_rnd_2, y_rnd_3, y_rnd_4, y_rnd_5))

                # Get randomized data to build trees - CLR and RCLR (PerMANOVA only)
                X_rnd_1_f, y_rnd_1_f = addcl2(X_train, y_train)
                X_rnd_2_f, y_rnd_2_f = addcl2(X_train, y_train)
                X_rnd_3_f, y_rnd_3_f = addcl2(X_train, y_train)
                X_rnd_4_f, y_rnd_4_f = addcl2(X_train, y_train)
                X_rnd_5_f, y_rnd_5_f = addcl2(X_train, y_train)
                X_rnd_f = np.vstack(
                    (X_rnd_1_f, X_rnd_2_f, X_rnd_3_f, X_rnd_4_f, X_rnd_5_f)
                )
                y_rnd_f = np.hstack(
                    (y_rnd_1_f, y_rnd_2_f, y_rnd_3_f, y_rnd_4_f, y_rnd_5_f)
                )

                #####################################################
                """
                Presence-Absence Transformation
                """
                do_pa = True
                print("Presence-Absence")
                if do_pa == True:
                    # Convert training, testing, and validation data to presence-absence
                    X_trn_pa = np.where(X_train_rare > 0, 1, 0)
                    X_tst_pa = np.where(X_tests_rare > 0, 1, 0)
                    X_rnd_pa = np.where(X_rnd > 0, 1, 0)

                    do_raw = True
                    if do_raw:

                        PER_data.append(
                            get_perm(
                                X_trn_pa,
                                y_train_rare,
                                "jaccard",
                                "Presence-Absence",
                                "Original Distances",
                                0,
                            )
                        )
                        print(PER_data[-1])
                        PER_data.append(
                            get_perm(
                                X_trn_pa,
                                y_train_rare,
                                "jaccard",
                                "Presence-Absence",
                                "PCoA",
                                1,
                            )
                        )
                        print(PER_data[-1])
                        PER_data.append(
                            get_perm(
                                X_trn_pa,
                                y_train_rare,
                                "jaccard",
                                "Presence-Absence",
                                "UMAP",
                                2,
                                8,
                            )
                        )
                        print(PER_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train_rare,
                                X_tst_pa,
                                y_tests_rare,
                                "None",
                                ExtraTreesClassifier(160),
                                "Presence-Absence",
                                "Original Data",
                                "Extra Trees",
                                0,
                            )
                        )
                        print(BAS_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train_rare,
                                X_tst_pa,
                                y_tests_rare,
                                "None",
                                LANDMarkClassifier(
                                    160,
                                    n_jobs=32,
                                    max_samples_tree=100,
                                    use_nnet=False,
                                ),
                                "Presence-Absence",
                                "Original Data",
                                "LANDMark",
                                0,
                            )
                        )
                        print(BAS_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train_rare,
                                X_tst_pa,
                                y_tests_rare,
                                "jaccard",
                                ExtraTreesClassifier(160),
                                "Presence-Absence",
                                "UMAP",
                                "Extra Trees",
                                1,
                                8,
                            )
                        )
                        print(BAS_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train_rare,
                                X_tst_pa,
                                y_tests_rare,
                                "jaccard",
                                LANDMarkClassifier(
                                    160,
                                    n_jobs=32,
                                    max_samples_tree=100,
                                    use_nnet=False,
                                ),
                                "Presence-Absence",
                                "UMAP",
                                "LANDMark",
                                1,
                                8,
                            )
                        )
                        print(BAS_data[-1])

                    do_et = True
                    if do_et:
                        # Train Unsupervised ETC model
                        """
                        1) Fit the model on the original and randomized data
                        2) Extract all terminal leaves
                        3) Create the co-occurance matrix (Equation 4)
                        4) Convert to dissimilarity (Equation 5)

                        Note, although we are using all the samples to get the leaves
                        only the training data was used to create the model.
                        """
                        et_unsup = ExtraTreesClassifier(160).fit(X_rnd_pa, y_rnd)
                        leaves = et_unsup.apply(X_trn_pa)
                        leaves_binary = OneHotEncoder(sparse=False).fit_transform(
                            leaves
                        )
                        S_xi_xj = np.dot(leaves_binary, leaves_binary.T)
                        S_xi_xj = S_xi_xj / 160
                        D_xi_xj = np.sqrt(1 - S_xi_xj).astype(np.float32)

                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train_rare,
                                "Learned",
                                "Presence-Absence",
                                "Unsupervised Extremely Randomized Trees",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            pcoa(
                                DistanceMatrix(D_xi_xj), number_of_dimensions=2
                            ).samples.values
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D,
                                y_train_rare,
                                "Learned",
                                "Presence-Absence",
                                "Unsupervised Extremely Randomized Trees (PCoA)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            UMAP(n_components=2, min_dist=0.001, metric="precomputed")
                            .fit_transform(D_xi_xj)
                            .astype(np.float32)
                        )
                        PER_data.append(
                            get_perm(
                                D,
                                y_train_rare,
                                "Learned",
                                "Presence-Absence",
                                "Unsupervised Extremely Randomized Trees (UMAP)",
                                3,
                                8,
                            )
                        )
                        print(PER_data[-1])

                        # Calculate Generalization Performance
                        # Step 1: Get leaves for test and train data, Encode Leaves
                        leaves_test = et_unsup.apply(X_tst_pa)
                        leaves_all = np.vstack((leaves, leaves_test))
                        leaves_trf = OneHotEncoder(sparse=False).fit(leaves_all)

                        leaves_train = leaves_trf.transform(leaves)
                        leaves_tests = leaves_trf.transform(leaves_test)

                        BAS_data.append(
                            get_classifer(
                                leaves_train,
                                y_train_rare,
                                leaves_tests,
                                y_tests_rare,
                                "hamming",
                                ExtraTreesClassifier(160),
                                "Presence-Absence",
                                "Extra Trees Embedding",
                                "Extra Trees",
                                1,
                                8,
                            )
                        )
                        print(BAS_data[-1])

                    do_lm = True
                    if do_lm:
                        et_unsup = LANDMarkClassifier(
                            160, n_jobs=32, max_samples_tree=100, use_nnet=False
                        ).fit(X_rnd_pa, y_rnd)
                        leaves = et_unsup.proximity(X_trn_pa)
                        D_xi_xj = pairwise_distances(leaves, metric="hamming").astype(
                            np.float32
                        )

                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train_rare,
                                "Learned",
                                "Presence-Absence",
                                "Unsupervised LANDMark",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            pcoa(
                                DistanceMatrix(D_xi_xj), number_of_dimensions=2
                            ).samples.values
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D,
                                y_train_rare,
                                "Learned",
                                "Presence-Absence",
                                "Unsupervised LANDMark (PCoA)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            UMAP(
                                n_components=2,
                                min_dist=0.001,
                                metric="precomputed",
                                n_neighbors=8,
                            )
                            .fit_transform(D_xi_xj)
                            .astype(np.float32)
                        )
                        PER_data.append(
                            get_perm(
                                D,
                                y_train_rare,
                                "Learned",
                                "Presence-Absence",
                                "Unsupervised LANDMark (UMAP)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        # Calculate Generalization Performance
                        leaves_train = leaves
                        leaves_tests = et_unsup.proximity(X_tst_pa)

                        BAS_data.append(
                            get_classifer(
                                leaves_train,
                                y_train_rare,
                                leaves_tests,
                                y_tests_rare,
                                "hamming",
                                LANDMarkClassifier(
                                    160, n_jobs=32, max_samples_tree=100, use_nnet=False
                                ),
                                "Presence-Absence",
                                "LANDMark Embedding",
                                "LANDMark",
                                1,
                                8,
                            )
                        )
                        print(BAS_data[-1])

                    do_to = True
                    if do_to:
                        # TreeOrdination distance matrix
                        et_unsup = TreeOrdination(
                            metric="hamming",
                            feature_names=feature_names,
                            unsup_n_estim=160,
                            n_iter_unsup=5,
                            n_jobs=32,
                            max_samples_tree=100,
                        ).fit(X_trn_pa, y_train_rare)
                        D_xi_xj = pairwise_distances(
                            et_unsup.R_PCA_emb, metric="euclidean"
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train_rare,
                                "Learned",
                                "Presence-Absence",
                                "TreeOrdination",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        p_result = et_unsup.predict(X_tst_pa)
                        bas_test = balanced_accuracy_score(y_tests_rare, p_result)
                        BAS_data.append(
                            (
                                "Presence-Absence",
                                "TreeOrdination Embedding",
                                "TreeOrdination",
                                "Learned",
                                bas_test,
                            )
                        )
                        print(BAS_data[-1])

                """
                Proportions and Bray-Curtis Transformation
                """
                do_bc = True
                print("Proportions and Bray-Curtis")
                if do_bc == True:
                    # Convert training, testing, and validation data to proportions
                    X_trn_pa = closure(X_train_rare)
                    X_tst_pa = closure(X_tests_rare)
                    X_rnd_pa = closure(X_rnd)

                    do_raw = True
                    if do_raw:

                        PER_data.append(
                            get_perm(
                                X_trn_pa,
                                y_train_rare,
                                "braycurtis",
                                "Proportions",
                                "Original Distances",
                                0,
                            )
                        )
                        print(PER_data[-1])
                        PER_data.append(
                            get_perm(
                                X_trn_pa,
                                y_train_rare,
                                "braycurtis",
                                "Proportions",
                                "PCoA",
                                1,
                            )
                        )
                        print(PER_data[-1])
                        PER_data.append(
                            get_perm(
                                X_trn_pa,
                                y_train_rare,
                                "braycurtis",
                                "Proportions",
                                "UMAP",
                                2,
                                8,
                            )
                        )
                        print(PER_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train_rare,
                                X_tst_pa,
                                y_tests_rare,
                                "None",
                                ExtraTreesClassifier(160),
                                "Proportions",
                                "Original Data",
                                "Extra Trees",
                                0,
                            )
                        )
                        print(BAS_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train_rare,
                                X_tst_pa,
                                y_tests_rare,
                                "None",
                                LANDMarkClassifier(
                                    160, n_jobs=32, max_samples_tree=100, use_nnet=False
                                ),
                                "Proportions",
                                "Original Data",
                                "LANDMark",
                                0,
                            )
                        )
                        print(BAS_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train_rare,
                                X_tst_pa,
                                y_tests_rare,
                                "braycurtis",
                                ExtraTreesClassifier(160),
                                "Bray-Curtis",
                                "UMAP",
                                "Extra Trees",
                                1,
                                8,
                            )
                        )
                        print(BAS_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train_rare,
                                X_tst_pa,
                                y_tests_rare,
                                "braycurtis",
                                LANDMarkClassifier(
                                    160, n_jobs=32, max_samples_tree=100, use_nnet=False
                                ),
                                "Bray-Curtis",
                                "UMAP",
                                "LANDMark",
                                1,
                                8,
                            )
                        )
                        print(BAS_data[-1])

                    do_et = True
                    if do_et:
                        # Train Unsupervised ETC model
                        """
                        1) Fit the model on the original and randomized data
                        2) Extract all terminal leaves
                        3) Create the co-occurance matrix (Equation 4)
                        4) Convert to dissimilarity (Equation 5)

                        Note, although we are using all the samples to get the leaves
                        only the training data was used to create the model.
                        """
                        et_unsup = ExtraTreesClassifier(160).fit(X_rnd_pa, y_rnd)
                        leaves = et_unsup.apply(X_trn_pa)
                        leaves_binary = OneHotEncoder(sparse=False).fit_transform(
                            leaves
                        )
                        S_xi_xj = np.dot(leaves_binary, leaves_binary.T)
                        S_xi_xj = S_xi_xj / 160
                        D_xi_xj = np.sqrt(1 - S_xi_xj).astype(np.float32)

                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train_rare,
                                "Learned",
                                "Proportions",
                                "Unsupervised Extremely Randomized Trees",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            pcoa(
                                DistanceMatrix(D_xi_xj), number_of_dimensions=2
                            ).samples.values
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D,
                                y_train_rare,
                                "Learned",
                                "Proportions",
                                "Unsupervised Extremely Randomized Trees (PCoA)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            UMAP(n_components=2, min_dist=0.001, metric="precomputed")
                            .fit_transform(D_xi_xj)
                            .astype(np.float32)
                        )
                        PER_data.append(
                            get_perm(
                                D,
                                y_train_rare,
                                "Learned",
                                "Proportions",
                                "Unsupervised Extremely Randomized Trees (UMAP)",
                                3,
                                8,
                            )
                        )
                        print(PER_data[-1])

                        # Calculate Generalization Performance
                        # Step 1: Get leaves for test and train data, Encode Leaves
                        leaves_test = et_unsup.apply(X_tst_pa)
                        leaves_all = np.vstack((leaves, leaves_test))
                        leaves_trf = OneHotEncoder(sparse=False).fit(leaves_all)

                        leaves_train = leaves_trf.transform(leaves)
                        leaves_tests = leaves_trf.transform(leaves_test)

                        BAS_data.append(
                            get_classifer(
                                leaves_train,
                                y_train_rare,
                                leaves_tests,
                                y_tests_rare,
                                "hamming",
                                ExtraTreesClassifier(160),
                                "Proportions",
                                "Extra Trees Embedding",
                                "Extra Trees",
                                1,
                                8,
                            )
                        )
                        print(BAS_data[-1])

                    do_lm = True
                    if do_lm:
                        et_unsup = LANDMarkClassifier(
                            160, n_jobs=32, max_samples_tree=100, use_nnet=False
                        ).fit(X_rnd_pa, y_rnd)
                        leaves = et_unsup.proximity(X_trn_pa)
                        D_xi_xj = pairwise_distances(leaves, metric="hamming").astype(
                            np.float32
                        )

                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train_rare,
                                "Learned",
                                "Proportions",
                                "Unsupervised LANDMark",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            pcoa(
                                DistanceMatrix(D_xi_xj), number_of_dimensions=2
                            ).samples.values
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D,
                                y_train_rare,
                                "Learned",
                                "Proportions",
                                "Unsupervised LANDMark (PCoA)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            UMAP(
                                n_components=2,
                                min_dist=0.001,
                                metric="precomputed",
                                n_neighbors=8,
                            )
                            .fit_transform(D_xi_xj)
                            .astype(np.float32)
                        )
                        PER_data.append(
                            get_perm(
                                D,
                                y_train_rare,
                                "Learned",
                                "Proportions",
                                "Unsupervised LANDMark (UMAP)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        # Calculate Generalization Performance
                        leaves_train = leaves
                        leaves_tests = et_unsup.proximity(X_tst_pa)

                        BAS_data.append(
                            get_classifer(
                                leaves_train,
                                y_train_rare,
                                leaves_tests,
                                y_tests_rare,
                                "hamming",
                                LANDMarkClassifier(
                                    160,
                                    n_jobs=32,
                                    max_samples_tree=100,
                                    use_nnet=False,
                                ),
                                "Proportions",
                                "LANDMark Embedding",
                                "LANDMark",
                                1,
                                8,
                            )
                        )
                        print(BAS_data[-1])

                    do_to = True
                    if do_to:
                        # TreeOrdination distance matrix
                        et_unsup = TreeOrdination(
                            metric="hamming",
                            feature_names=feature_names,
                            unsup_n_estim=160,
                            n_iter_unsup=5,
                            n_jobs=32,
                            max_samples_tree=100,
                            scale=True,
                        ).fit(X_train_rare, y_train_rare)
                        D_xi_xj = pairwise_distances(
                            et_unsup.R_PCA_emb, metric="euclidean"
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train_rare,
                                "Learned",
                                "Proportions",
                                "TreeOrdination",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        p_result = et_unsup.predict(X_tst_pa)
                        bas_test = balanced_accuracy_score(y_tests_rare, p_result)
                        BAS_data.append(
                            (
                                "Proportions",
                                "TreeOrdination Embedding",
                                "TreeOrdination",
                                "Learned",
                                bas_test,
                            )
                        )
                        print(BAS_data[-1])

                """
                CLR Transformation
                """
                do_clr = True
                print("Centered Log-Ratio")
                if do_clr == True:
                    # Convert training, testing, and validation data to presence-absence
                    X_trn_pa = clr(multiplicative_replacement(closure(X_train)))
                    X_tst_pa = clr(multiplicative_replacement(closure(X_tests)))
                    X_rnd_pa = clr(multiplicative_replacement(closure(X_rnd_f)))

                    do_raw = True
                    if do_raw:

                        PER_data.append(
                            get_perm(
                                X_trn_pa,
                                y_train,
                                "euclidean",
                                "Centered Log-Ratio",
                                "Original Distances",
                                0,
                            )
                        )
                        print(PER_data[-1])
                        PER_data.append(
                            get_perm(
                                X_trn_pa,
                                y_train,
                                "euclidean",
                                "Centered Log-Ratio",
                                "PCoA",
                                1,
                            )
                        )
                        print(PER_data[-1])
                        PER_data.append(
                            get_perm(
                                X_trn_pa,
                                y_train,
                                "euclidean",
                                "Centered Log-Ratio",
                                "UMAP",
                                2,
                                8,
                            )
                        )
                        print(PER_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train,
                                X_tst_pa,
                                y_tests,
                                "None",
                                ExtraTreesClassifier(160),
                                "Centered Log-Ratio",
                                "Original Data",
                                "Extra Trees",
                                0,
                            )
                        )
                        print(BAS_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train,
                                X_tst_pa,
                                y_tests,
                                "None",
                                LANDMarkClassifier(
                                    160,
                                    n_jobs=32,
                                    max_samples_tree=100,
                                    use_nnet=False,
                                ),
                                "Centered Log-Ratio",
                                "Original Data",
                                "LANDMark",
                                0,
                            )
                        )
                        print(BAS_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train,
                                X_tst_pa,
                                y_tests,
                                "euclidean",
                                ExtraTreesClassifier(160),
                                "Centered Log-Ratio",
                                "UMAP",
                                "Extra Trees",
                                1,
                                8,
                            )
                        )
                        print(BAS_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train,
                                X_tst_pa,
                                y_tests,
                                "euclidean",
                                LANDMarkClassifier(
                                    160,
                                    n_jobs=32,
                                    max_samples_tree=100,
                                    use_nnet=False,
                                ),
                                "Centered Log-Ratio",
                                "UMAP",
                                "LANDMark",
                                1,
                                8,
                            )
                        )
                        print(BAS_data[-1])

                    do_et = True
                    if do_et:
                        # Train Unsupervised ETC model
                        """
                        1) Fit the model on the original and randomized data
                        2) Extract all terminal leaves
                        3) Create the co-occurance matrix (Equation 4)
                        4) Convert to dissimilarity (Equation 5)

                        Note, although we are using all the samples to get the leaves
                        only the training data was used to create the model.
                        """
                        et_unsup = ExtraTreesClassifier(160).fit(X_rnd_pa, y_rnd_f)
                        leaves = et_unsup.apply(X_trn_pa)
                        leaves_binary = OneHotEncoder(sparse=False).fit_transform(
                            leaves
                        )
                        S_xi_xj = np.dot(leaves_binary, leaves_binary.T)
                        S_xi_xj = S_xi_xj / 160
                        D_xi_xj = np.sqrt(1 - S_xi_xj).astype(np.float32)

                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train,
                                "Learned",
                                "Centered Log-Ratio",
                                "Unsupervised Extremely Randomized Trees",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            pcoa(
                                DistanceMatrix(D_xi_xj), number_of_dimensions=2
                            ).samples.values
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D,
                                y_train,
                                "Learned",
                                "Centered Log-Ratio",
                                "Unsupervised Extremely Randomized Trees (PCoA)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            UMAP(n_components=2, min_dist=0.001, metric="precomputed")
                            .fit_transform(D_xi_xj)
                            .astype(np.float32)
                        )
                        PER_data.append(
                            get_perm(
                                D,
                                y_train,
                                "Learned",
                                "Centered Log-Ratio",
                                "Unsupervised Extremely Randomized Trees (UMAP)",
                                3,
                                8,
                            )
                        )
                        print(PER_data[-1])

                        # Calculate Generalization Performance
                        # Step 1: Get leaves for test and train data, Encode Leaves
                        leaves_test = et_unsup.apply(X_tst_pa)
                        leaves_all = np.vstack((leaves, leaves_test))
                        leaves_trf = OneHotEncoder(sparse=False).fit(leaves_all)

                        leaves_train = leaves_trf.transform(leaves)
                        leaves_tests = leaves_trf.transform(leaves_test)

                        BAS_data.append(
                            get_classifer(
                                leaves_train,
                                y_train,
                                leaves_tests,
                                y_tests,
                                "hamming",
                                ExtraTreesClassifier(160),
                                "Centered Log-Ratio",
                                "Extra Trees Embedding",
                                "Extra Trees",
                                1,
                                8,
                            )
                        )
                        print(BAS_data[-1])

                    do_lm = True
                    if do_lm:
                        et_unsup = LANDMarkClassifier(
                            160, n_jobs=32, max_samples_tree=100, use_nnet=False
                        ).fit(X_rnd_pa, y_rnd_f)
                        leaves = et_unsup.proximity(X_trn_pa)
                        D_xi_xj = pairwise_distances(leaves, metric="hamming").astype(
                            np.float32
                        )

                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train,
                                "Learned",
                                "Centered Log-Ratio",
                                "Unsupervised LANDMark",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            pcoa(
                                DistanceMatrix(D_xi_xj), number_of_dimensions=2
                            ).samples.values
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D,
                                y_train,
                                "Learned",
                                "Centered Log-Ratio",
                                "Unsupervised LANDMark (PCoA)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            UMAP(
                                n_components=2,
                                min_dist=0.001,
                                metric="precomputed",
                                n_neighbors=8,
                            )
                            .fit_transform(D_xi_xj)
                            .astype(np.float32)
                        )
                        PER_data.append(
                            get_perm(
                                D,
                                y_train,
                                "Learned",
                                "Centered Log-Ratio",
                                "Unsupervised LANDMark (UMAP)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        # Calculate Generalization Performance
                        leaves_train = leaves
                        leaves_tests = et_unsup.proximity(X_tst_pa)

                        BAS_data.append(
                            get_classifer(
                                leaves_train,
                                y_train,
                                leaves_tests,
                                y_tests,
                                "hamming",
                                LANDMarkClassifier(
                                    160,
                                    n_jobs=32,
                                    max_samples_tree=100,
                                    use_nnet=False,
                                ),
                                "Centered Log-Ratio",
                                "LANDMark Embedding",
                                "LANDMark",
                                1,
                                8,
                            )
                        )

                        print(BAS_data[-1])

                    do_to = True
                    if do_to:
                        # TreeOrdination distance matrix
                        et_unsup = TreeOrdination(
                            metric="hamming",
                            feature_names=feature_names,
                            unsup_n_estim=160,
                            n_iter_unsup=1,
                            n_jobs=32,
                            max_samples_tree=100,
                            n_neighbors=8,
                            clr_trf=True,
                        ).fit(X_train, y_train)
                        D_xi_xj = pairwise_distances(
                            et_unsup.R_PCA_emb, metric="euclidean"
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train,
                                "Learned",
                                "Centered Log-Ratio",
                                "TreeOrdination",
                                3,
                            )
                        )

                        p_result = et_unsup.predict(X_tests)
                        bas_test = balanced_accuracy_score(y_tests, p_result)
                        BAS_data.append(
                            (
                                "Centered Log-Ratio",
                                "TreeOrdination Embedding",
                                "TreeOrdination",
                                "Learned",
                                bas_test,
                            )
                        )

                        print(BAS_data[-1])
                        print(PER_data[-1])

                """
                RCLR Transformation
                """
                do_rclr = True
                print("Robust Centered Log-Ratio")
                if do_rclr == True:

                    X_prop_train = np.copy(X_train, "C")
                    X_prop_train = rclr(X_prop_train.transpose()).transpose()
                    M = MatrixCompletion(2, max_iterations=1000).fit(X_prop_train)
                    X_trn_pa = M.U
                    D = M.distance

                    X_rnd_pa = np.copy(X_rnd_f, "C")
                    X_rnd_pa = rclr(X_rnd_pa.transpose()).transpose()
                    M = MatrixCompletion(2, max_iterations=1000).fit(X_rnd_pa)
                    X_rnd_pa = M.solution

                    do_raw = True
                    if do_raw:

                        PER_data.append(
                            get_perm(
                                D,
                                y_train,
                                "euclidean",
                                "Robust Centered Log-Ratio",
                                "Original Distances",
                                3,
                            )
                        )
                        print(PER_data[-1])
                        PER_data.append(
                            get_perm(
                                X_trn_pa,
                                y_train,
                                "euclidean",
                                "Robust Centered Log-Ratio",
                                "PCoA",
                                1,
                            )
                        )
                        print(PER_data[-1])
                        PER_data.append(
                            get_perm(
                                D,
                                y_train,
                                "precomputed",
                                "Robust Centered Log-Ratio",
                                "UMAP",
                                2,
                                8,
                            )
                        )
                        print(PER_data[-1])

                    do_et = True
                    if do_et:
                        # Train Unsupervised ETC model
                        """
                        1) Fit the model on the original and randomized data
                        2) Extract all terminal leaves
                        3) Create the co-occurance matrix (Equation 4)
                        4) Convert to dissimilarity (Equation 5)

                        Note, although we are using all the samples to get the leaves
                        only the training data was used to create the model.
                        """
                        et_unsup = ExtraTreesClassifier(160).fit(X_rnd_pa, y_rnd_f)
                        leaves = et_unsup.apply(X_rnd_pa[0 : X_trn_pa.shape[0]])
                        leaves_binary = OneHotEncoder(sparse=False).fit_transform(
                            leaves
                        )
                        S_xi_xj = np.dot(leaves_binary, leaves_binary.T)
                        S_xi_xj = S_xi_xj / 160
                        D_xi_xj = np.sqrt(1 - S_xi_xj).astype(np.float32)

                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train,
                                "Learned",
                                "Robust Centered Log-Ratio",
                                "Unsupervised Extremely Randomized Trees",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            pcoa(
                                DistanceMatrix(D_xi_xj), number_of_dimensions=2
                            ).samples.values
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D,
                                y_train,
                                "Learned",
                                "Robust Centered Log-Ratio",
                                "Unsupervised Extremely Randomized Trees (PCoA)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            UMAP(n_components=2, min_dist=0.001, metric="precomputed")
                            .fit_transform(D_xi_xj)
                            .astype(np.float32)
                        )
                        PER_data.append(
                            get_perm(
                                D,
                                y_train,
                                "Learned",
                                "Robust Centered Log-Ratio",
                                "Unsupervised Extremely Randomized Trees (UMAP)",
                                3,
                                8,
                            )
                        )
                        print(PER_data[-1])

                    do_lm = True
                    if do_lm:
                        et_unsup = LANDMarkClassifier(
                            160, n_jobs=32, max_samples_tree=100, use_nnet=False
                        ).fit(X_rnd_pa, y_rnd_f)
                        leaves = et_unsup.proximity(X_rnd_pa[0 : X_trn_pa.shape[0]])
                        D_xi_xj = pairwise_distances(leaves, metric="hamming").astype(
                            np.float32
                        )

                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train,
                                "Learned",
                                "Robust Centered Log-Ratio",
                                "Unsupervised LANDMark",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            pcoa(
                                DistanceMatrix(D_xi_xj), number_of_dimensions=2
                            ).samples.values
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D,
                                y_train,
                                "Learned",
                                "Robust Centered Log-Ratio",
                                "Unsupervised LANDMark (PCoA)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            UMAP(
                                n_components=2,
                                min_dist=0.001,
                                metric="precomputed",
                                n_neighbors=8,
                            )
                            .fit_transform(D_xi_xj)
                            .astype(np.float32)
                        )
                        PER_data.append(
                            get_perm(
                                D,
                                y_train,
                                "Learned",
                                "Robust Centered Log-Ratio",
                                "Unsupervised LANDMark (UMAP)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                    do_to = True
                    if do_to:
                        # TreeOrdination distance matrix
                        et_unsup = TreeOrdination(
                            metric="hamming",
                            feature_names=feature_names,
                            unsup_n_estim=160,
                            n_iter_unsup=5,
                            n_jobs=32,
                            max_samples_tree=100,
                            n_neighbors=8,
                            rclr_trf=True,
                        ).fit(X_train, y_train)
                        D_xi_xj = pairwise_distances(
                            et_unsup.R_PCA_emb, metric="euclidean"
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train,
                                "Learned",
                                "Robust Centered Log-Ratio",
                                "TreeOrdination",
                                3,
                            )
                        )
                        print(PER_data[-1])

            BAS_data = pd.DataFrame(
                BAS_data,
                columns=[
                    "Transformation",
                    "Comparision Type",
                    "Model",
                    "Metric",
                    "BACC",
                ],
            )
            BAS_data.to_csv("%s_BACC.csv" % experiment[dataset_type])

            PER_data = pd.DataFrame(
                PER_data,
                columns=[
                    "Transformation",
                    "Comparision Type",
                    "Metric",
                    "Pseudo-F",
                    "p-value",
                    "R-Squared",
                ],
            )
            PER_data.to_csv("%s_PerMANOVA.csv" % experiment[dataset_type])

    feature_importance_test = False
    if feature_importance_test:
        # Feature names
        feature_names = np.asarray(cluster_name)

        # Create a test-train split (index only)
        X_train, X_test, y_train, y_test = train_test_split(
            X_signal.values,
            meta["Host_disease"].values,
            train_size=0.7,
            random_state=0,
            stratify=meta["Host_disease"].values,
        )

        # Retain ASVs present in at least 3 samples
        X_sum = np.where(X_train > 0, 1, 0)
        X_sum = np.sum(X_sum, axis=0)
        removed_ASVs = np.where(X_sum >= 3, True, False)

        X_train = X_train[:, removed_ASVs]

        X_test = X_test[:, removed_ASVs]

        feature_names = feature_names[removed_ASVs]

        # Train the model
        model = TreeOrdination(
            metric="hamming",
            feature_names=feature_names,
            unsup_n_estim=160,
            n_iter_unsup=5,
            n_jobs=10,
            n_neighbors=8,
            clr_trf=True,
            max_samples_tree=100,
        ).fit(X_train, y_train)

        # Plot data
        test_emb_true = model.emb_transform(X_test)
        test_emb = model.approx_emb(X_test)

        sns.scatterplot(x=test_emb[:, 0], y=test_emb[:, 1], hue=y_test)

        pc1 = model.R_PCA.explained_variance_ratio_[0] * 100
        pc2 = model.R_PCA.explained_variance_ratio_[1] * 100
        plt.xlabel("PCA 1 (%.3f Percent)" % pc1)
        plt.ylabel("PCA 2 (%.3f Percent)" % pc2)
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig("TreeOrd_Proj_CD_HC.svg")
        plt.close()

        # Get PerMANOVA of projection
        perm_res = get_perm(test_emb, y_test, "euclidean", "TreeOrd", "TreeOrd", 0)
        print(perm_res)

        # Get SHAP Scores
        X_test_clr = model.scale_clr(X_test)

        predictive_model = model.l_model
        mse_test = mean_squared_error(test_emb, test_emb_true)
        print(mse_test)

        E = sh.Explainer(predictive_model, feature_names=feature_names)
        shap_test = E(X_test_clr)

        sh.plots.bar(
            shap_test[:, :, 0].cohorts(y_test).abs.mean(0), max_display=15, show=False
        )

        plt.show()

    get_stats_perm = True
    if get_stats_perm:
        df = pd.read_csv("Crohns_Disease_PerMANOVA.csv")

        df["Metric"] = [x.capitalize() for x in df["Metric"].values]

        order = [
            "Original Distances",
            "PCoA",
            "UMAP",
            "Unsupervised Extremely Randomized Trees",
            "Unsupervised Extremely Randomized Trees (PCoA)",
            "Unsupervised Extremely Randomized Trees (UMAP)",
            "Unsupervised LANDMark",
            "Unsupervised LANDMark (PCoA)",
            "Unsupervised LANDMark (UMAP)",
            "TreeOrdination",
        ]

        col_order = [
            "Presence-Absence",
            "Proportions",
            "Centered Log-Ratio",
            "Robust Centered Log-Ratio",
        ]

        g = sns.catplot(
            data=df,
            x="Comparision Type",
            y="p-value",  # Or Pseudo-F
            hue="Metric",
            col="Transformation",
            kind="bar",
            ci=95,
            order=order,
            col_order=col_order,
            n_boot=2000,
            dodge=False,
        )

        g.set_xticklabels(rotation=90)

        plt.tight_layout()

        plt.show()
        plt.close()

    get_stats_bacc = False
    if get_stats_bacc:
        from itertools import combinations
        from statannotations.Annotator import Annotator

        df = pd.read_csv("Crohns_Disease_BACC.csv")

        df["Transformation"] = np.where(
            df["Transformation"] == "Bray-Curtis", "Proportions", df["Transformation"]
        )
        df["Metric"] = np.where(df["Metric"] == "hamming", "Learned", df["Metric"])

        df["Metric"] = [x.capitalize() for x in df["Metric"].values]

        df["Model (Metric)"] = [
            "%s (%s)" % (x, df["Metric"].values[i])
            for i, x in enumerate(df["Model"].values)
        ]

        # Set Figure and Axes
        fig, ax = plt.subplots(nrows=1, ncols=3)

        # Statistical analysis preparation
        order = [
            "Extra Trees (None)",
            "LANDMark (None)",
            "Extra Trees (Jaccard)",
            "LANDMark (Jaccard)",
            "Extra Trees (Learned)",
            "LANDMark (Learned)",
            "TreeOrdination (Learned)",
        ]
        pairs = list(combinations(order, 2))
        df_ss = np.where(df["Transformation"] == "Presence-Absence", True, False)
        sns.barplot(
            data=df[df_ss],
            x="Model (Metric)",
            y="BACC",
            hue="Comparision Type",
            ci=95,
            n_boot=2000,
            dodge=False,
            ax=ax[0],
        )
        ax[0].set_title("Presence-Absence")
        ax[0].get_legend().remove()
        for tick in ax[0].get_xticklabels():
            tick.set_rotation(90)

        annotator = Annotator(
            ax[0], pairs, data=df[df_ss], x="Model (Metric)", y="BACC"
        )
        annotator.configure(
            test="Wilcoxon",
            text_format="star",
            comparisons_correction="fdr_bh",
            loc="outside",
            correction_format="replace",
        )
        annotator.apply_and_annotate()

        order = [
            "Extra Trees (None)",
            "LANDMark (None)",
            "Extra Trees (Braycurtis)",
            "LANDMark (Braycurtis)",
            "Extra Trees (Learned)",
            "LANDMark (Learned)",
            "TreeOrdination (Learned)",
        ]
        pairs = list(combinations(order, 2))
        df_ss = np.where(df["Transformation"] == "Proportions", True, False)
        sns.barplot(
            data=df[df_ss],
            x="Model (Metric)",
            y="BACC",
            hue="Comparision Type",
            ci=95,
            n_boot=2000,
            dodge=False,
            ax=ax[1],
        )
        ax[1].set_title("Proportion")
        ax[1].get_legend().remove()
        for tick in ax[1].get_xticklabels():
            tick.set_rotation(90)

        annotator = Annotator(
            ax[1], pairs, data=df[df_ss], x="Model (Metric)", y="BACC"
        )
        annotator.configure(
            test="Wilcoxon",
            text_format="star",
            comparisons_correction="fdr_bh",
            loc="outside",
            correction_format="replace",
        )
        annotator.apply_and_annotate()

        order = [
            "Extra Trees (None)",
            "LANDMark (None)",
            "Extra Trees (Euclidean)",
            "LANDMark (Euclidean)",
            "Extra Trees (Learned)",
            "LANDMark (Learned)",
            "TreeOrdination (Learned)",
        ]
        pairs = list(combinations(order, 2))
        df_ss = np.where(df["Transformation"] == "Centered Log-Ratio", True, False)
        sns.barplot(
            data=df[df_ss],
            x="Model (Metric)",
            y="BACC",
            hue="Comparision Type",
            ci=95,
            n_boot=2000,
            dodge=False,
            ax=ax[2],
        )
        ax[2].set_title("Centered Log-Ratio")
        for tick in ax[2].get_xticklabels():
            tick.set_rotation(90)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="center left")

        annotator = Annotator(
            ax[2], pairs, data=df[df_ss], x="Model (Metric)", y="BACC"
        )
        annotator.configure(
            test="Wilcoxon",
            text_format="star",
            comparisons_correction="fdr_bh",
            loc="outside",
            correction_format="replace",
        )
        annotator.apply_and_annotate()
