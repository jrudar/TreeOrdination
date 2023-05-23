## Overview of Section

This section provides an overview of the `TreeOrdination` and the different parameters
of the `TreeOrdination` class and its methods.

## Class

    class TreeOrdination.TreeOrdination(feature_names, resampler =  NoResample(),  metric  = "hamming", supervised_clf =  ExtraTreesClassifier(1024),
                                        proxy_model = ExtraTreesRegressor(1024), n_iter_unsup = 5, unsup_n_estim = 160, transformer =  NoTransform(),
                                        exclude_col = [False, 0],  n_neighbors = 8, n_components  = 2, min_dist = 0.001, n_jobs = 4)

### Parameters

    feature_names: list-like, required
        A list of feature names.

    resampler: default = NoResample
        The re-sampling  method to be used. Should follow the format
        used by 'imbalaced-learn'. Use this to control how many samples
        are passed to LANDMark.
    
    metric: str, default = "hamming"
        The metric used by UMAP to calculate the dissimilarity between 
        LANDMark embeddings.
        
    supervised_clf: default = ExtraTreesClassifier(1024)
        The classification model used to predict the class of each sample
        using the unsupervised projections.
        
    proxy_model: default = ExtraTreesClassifier(1024)
        The regression model used to predict the location of each sample
        in the projected space.

    landmark_model: default = LANDMarkClassifier(160, use_nnet = False, n_jobs = 8)

    n_iter_unsup: int, default = 5
        The number of LANDMark embeddings which will be used to construct
        the final embedding.
                
    transformer: default = NoTransform()
        The pre-processing method used to transform data. Should follow
        the 'scikit-learn' format.
        
    exclude_col: list-like, default = [False, [0]]
        Specifies which columns should be excluded for scaling and/or
        transformation. If the first entry in the list is true the columns
        specified by the second entry will be excluded from scaling.
        
    n_neighbors: int, default = 8
        The 'n_neighbors' parameter of UMAP. A larger value will capture
        more of the global structure of the data while smaller values will
        focus more on the local structure of the data. Larger datasets will
        likely need a larger value for this parameter.
     
    n_components: int, default = 2
        The number of components of the final unsupervised projection.
     
    min_dist: float, default = 0.001
        The 'min_dist' parameter of UMAP.

### Attributes

    encoded_labels: LabelEncoder
        A label encoder used to transform class labels.

    X, y: np.ndarray
        A saved copy of the training data.

    f_name: np.ndarray
        The feature names

    p_model: 
        A classifier trained on R_final and the encoded class labels (y)

    proj_transformer: 
        A pre-processing transformer used to transfrom the
        the training data, X.

    l_model: 
        A regression model which maps training data to the output produced
        by R_PCA (see below).

    Rs: list
        A list of trained LANDMark models

    LM_emb: np.ndarray
        A list of LANDMark proximities

    transformers: list
        A list of pre-processing transformers to be applied prior to fitting
        a LANDMark model or using a LANDMark model to calculate proximities.

    UMAP_trf: UMAP
        A 'UMAP' transformer trained using R_final.

    UMAP_emb: np.ndarray
        The UMAP transformed data..

    PCA_trf: PCA
        A 'scikit-learn' PCA transformer fit using the output of the UMAP
        transformer.

    PCA_emb: np.ndarray
        The PCA transformed data.

### Methods

    fit(X, y, **fit_params)
        Fits a `TreeOrdination` model.

        Parameters:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        y: NumPy array of shape (m,) where 'm' is the number of samples. Each entry
        of 'y' should be a factor.

        Returns:

        A fitted TreeOrdination object.

    get_importance(X, y = None)
        Prepares an 'alibi' Explainer object.

        Parameters:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        y: NumPy array of shape (m,) where 'm' is the number of samples. Each entry
        of 'y' should be a factor.

        Returns:

        A fitted Explainer object.

    plot_importance_global(X, y, class_name, n_features = 10, axis = 0, summary = "median", **kwargs)
        Produces a plot of per-class global feature importance scores.

        Parameters:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        y: NumPy array of shape (m,) where 'm' is the number of samples. Each entry
        of 'y' should be a factor.

        class_name: str
            The name of the class for which importance scores will be plotted.

        n_features: int, default = 10
            The number of features to plot

        axis: int, default = 0
            The axis of interest from the R_PCA projection

        summary: str, default = "median"
            The method used to summarize Shapley values.

        Returns:

        A 'matplotlib' figure and axis object

    plot_importance_local(X, n_features = 10, axis = 0, **kwargs)
        Produces a plot of feature importance scores on a per-sample level.

        Parameters:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        n_features: int, default = 10
            The number of features to plot

        axis: int, default = 0
            The axis of interest from the R_PCA projection

        Returns:

        A 'matplotlib' figure and axis object

    plot_projection(X, y, ax_1 = 0, ax_2 = 0, use_approx = True, **kwargs)
        Produces a plot of feature importance scores on a per-sample level.

        Parameters:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        y: NumPy array of shape (m,) where 'm' is the number of samples. Each entry
        of 'y' should be a factor.

        ax_1: int, default = 0
            The first axis of interest from the R_PCA projection

        ax_2: int, default = 1
            The second axis of interest from the R_PCA projection

        use_approx: bool, default = True
            Whether the proxy model (l_model from above) or full model will be used
            to create the plot for the projection.

        Returns:

        A 'matplotlib' figure and axis object

    predict_proba(X)
        Predicts class probabilities.

        Input:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        Returns:

        A np.ndarray of shape (m, p) where 'm' is the number of samples in X and
        'p' is the number of classes.

    emb_transform(X, y, trf_type)
        Transforms X into a lower dimensional embedding.

        Parameters:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        trf_type: str, default = "PCA"
        Transforms data using the specified transformer. Options are "approx", 
        "LM", "UMAP", or "PCA".

        Returns:

        X_transformed: NumPy array of shape (m, p) where 'm' is the number of samples and 'p'
        the number of components specified at the initialization of `TreeOrdination`.

## Class

    class TreeOrdination.CLRClosureTransform(do_clr = False, delta = None)

### Parameters (For CLRClosureTransform() Only)

    do_clr: bool, default = False
        Applies the CLR transformation if True. Otherwise only the closure
        operation is applied.

### Methods (CLRClosureTransform)

    fit_transform(X, y = None, **kwargs)
        Fits a CLRClosureTransform/NoTransform model and returns the transformed data.
        A NoTransform model will just return the original data.

        Parameters:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        Returns:

        X_transformed: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features.

    transform(X, y = None, **kwargs)
        Takes a CLRClosureTransform/NoTransform model and returns the transformed data.
        A NoTransform model will just return the original data.

        Parameters:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        Returns:

        X_transformed: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features.