### TreeOrdination
Implementation of a wrapper which creates unsupervised projections using LANDMark and UMAP.

### Future Work
In the future we hope to add the following feature to TreeOrdination:
    1) A simple interface to apply normalization and standardization procedures to the dataset.
    2) A simple interface to apply useful transformations to the dataset.
    3) The ability to exclude columns from normalization/standardization/transformation.

### Install
The LANDMark package is needed for TreeOrdination to work. It is available at: https://github.com/jrudar/LANDMark

Once downloaded, go to the TreeOrdination directory and type:
    pip install .
    
### Class Parameters
    The current hyper-parameters are available for tuning.

    feature_names: list-like, required
        A list of feature names.

    resample_on_y: Currently not used.
    
    metric: str, default = "hamming"
        The metric used by UMAP to calculate the dissimilarity between 
        LANDMark embeddings.
        
    supervised_clf: default = ExtraTreesClassifier(1024)
        The classification model used to predict the class of each sample
        using the unsupervised projections.
        
    n_iter_unsup: int, default = 5
        The number of LANDMark embeddings which will be used to construct
        the final embedding.
        
    unsup_n_estim: int, default = 160
        The number of decision trees in each LANDMark classifier.
        
    n_jobs: int, default = 4
        The number of processes used by LANDMark to train each classifier.
        
    n_neighbors: int, default = 8
        The 'n_neighbors' parameter of UMAP. A larger value will capture
        more of the global structure of the data while smaller values will
        focus more on the local structure of the data. Larger datasets will
        likely need a larger value for this parameter.
        
    max_samples_tree: int, default = 100
        Specifies how many samples will be used to train each LANDMark tree.
        
    min_dist: float, default = 0.001
        The 'min_dist' parameter of UMAP.
        
    scale: bool, default = False
        Specifies if each row in X should be divided by its sum.
        
    clr_trf: bool, default = False
        Specifies if the data should be center log-ratio transformed.
        
    rclr_trf: bool, default = False
        Specifies if the data should be transformed using the robust centered
        log-ratio transformation.
        
    exclude_col: list-like, default = [False, 0]
        Specifies which columns should be excluded for scaling and/or
        transformation. Currently, ff the first entry in the list is true
        only the last column in X is excluded from transformation. This
        behavior will change in the future so that the second entry
        in this list will specify a list of columns to exclude.
        
    n_components: int, default = 2
        The number of components of the final unsupervised projection.
            
### Fit Parameters
        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (features, taxa, OTUs, ASVs, etc).

        y: NumPy array of shape (m,) where 'm' is the number of samples. Each entry
        of 'y' should be a factor.
        
### Example Usage
        from TreeOrdination import TreeOrdination
        from sklearn.datasets import make_classification
        
        #Create the dataset
        X, y = make_classification(n_samples = 200, n_informative = 20)
        
        #Give features a name
        f_names = ["Feature %s" %str(i) for i in range(X.shape[0])]
        
        tree_ord = TreeOrdination(feature_names = f_names).fit(X, y)

        #This is the LANDMark embedding of the dataset. This dataset is used to train the supervised model ('supervised_clf' parameter)
        landmark_embedding = tree_ord.R_final
        
        #This is the UMAP projection of the LANDMark embedding
        umap_projection = tree_ord.tree_emb
        
        #This is the PCA projetion of the UMAP embedding
        pca_projection = tree_ord.R_PCA_emb      

### References

    Rudar, J., Porter, T.M., Wright, M., Golding G.B., Hajibabaei, M. LANDMark: an ensemble 
    approach to the supervised selection of biomarkers in high-throughput sequencing data. 
    BMC Bioinformatics 23, 110 (2022). https://doi.org/10.1186/s12859-022-04631-z

    Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: 
    Machine Learning in Python. Journal of Machine Learning Research. 2011;12:2825–30. 
   
    Geurts P, Ernst D, Wehenkel L. Extremely Randomized Trees. Machine Learning. 2006;63(1):3–42.
    
    Rudar J, Golding G.B., Kremer S.C., Hajibabaei M. Decision Tree Ensembles Utilizing 
    Multivariate Splits Are Effective at Investigating Beta-Diversity in Medically 
    Relevant 16S Amplicon Sequencing Data. bioRxiv 2022.03.31.486647; 
    doi: https://doi.org/10.1101/2022.03.31.486647

