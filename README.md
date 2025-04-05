### TreeOrdination
[![CI](https://github.com/jrudar/TreeOrdination/actions/workflows/ci.yml/badge.svg)](https://github.com/jrudar/TreeOrdination/actions/workflows/ci.yml)

Implementation of a wrapper which creates unsupervised projections using LANDMark and UMAP.
    
### Install
From PyPI:

```
pip install TreeOrdination
```

From source:

```bash
git clone https://github.com/jrudar/TreeOrdination.git
cd TreeOrdination
pip install .
# or create a virtual environment
python -m venv venv
source venv/bin/activate
pip install .
```
            
### Example Usage
        from TreeOrdination import TreeOrdination
        from sklearn.datasets import make_classification
        
        #Create the dataset
        X, y = make_classification(n_samples = 200, n_informative = 20)
        
        #Give features a name
        f_names = ["Feature %s" %str(i) for i in range(X.shape[0])]
        
        tree_ord = TreeOrdination(feature_names = f_names).fit(X, y)

        #This is the LANDMark embedding of the dataset. This dataset is used to train the supervised model ('supervised_clf' parameter)
        landmark_embedding = tree_ord.LM_emb
        
        #This is the UMAP projection of the LANDMark embedding
        umap_projection = tree_ord.UMAP_emb
        
        #This is the PCA projetion of the UMAP embedding
        pca_projection = tree_ord.PCA_emb     

### Notebooks and Other Examples
Comming Soon.
When available, examples of how to use `TreeOrdination` will be found [here](notebooks/README.md).

### Interface
An overview of the API can be found [here](docs/API.md).

### Contributing
To contribute to the development of `TreeOrdination` please read our [contributing guide](docs/CONTRIBUTING.md)

### References

Rudar, J., Porter, T.M., Wright, M., Golding G.B., Hajibabaei, M. LANDMark: an ensemble 
approach to the supervised selection of biomarkers in high-throughput sequencing data. 
BMC Bioinformatics 23, 110 (2022). https://doi.org/10.1186/s12859-022-04631-z

Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: 
Machine Learning in Python. Journal of Machine Learning Research. 2011;12:2825–30. 
   
Geurts P, Ernst D, Wehenkel L. Extremely Randomized Trees. Machine Learning. 2006;63(1):3–42.
    
Rudar, J., Golding, G.B., Kremer, S.C., Hajibabaei, M. (2023). Decision Tree Ensembles Utilizing 
Multivariate Splits Are Effective at Investigating Beta Diversity in Medically Relevant 16S Amplicon 
Sequencing Data. Microbiology Spectrum e02065-22.

Jai Ram Rideout, Greg Caporaso, Evan Bolyen, Daniel McDonald, Yoshiki Vázquez Baeza, Jorge Cañardo
Alastuey, Anders Pitman, Jamie Morton, Qiyun Zhu, Jose Navas, Kestrel Gorlick, Justine Debelius, 
Zech Xu, Matt Aton, llcooljohn, Joshua Shorenstein, Laurent Luce, Will Van Treuren, John Chase, 
… Dr. K. D. Murray. (2025). scikit-bio/scikit-bio: scikit-bio 0.6.3 (0.6.3). 
Zenodo. https://doi.org/10.5281/zenodo.14640761

