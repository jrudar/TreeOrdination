from setuptools import setup

setup(name="TreeOrdination",
                 version="1.0.2",
                 author="Josip Rudar, G. Brian Golding, Stefan C. Kremer, Mehrdad Hajibabaei",
                 author_email="rudarj@uoguelph.ca",
                 description="Decision Tree Ensembles Utilizing Multivariate Splits Are Effective at Investigating Beta-Diversity in Medically Relevant 16S Amplicon Sequencing Data",
                 url="https://github.com/jrudar/TreeOrdination",
                 license = "MIT",
                 keywords = "biomarker selection, metagenomics, metabarcoding, biomonitoring, ecological assessment, machine learning, supervised learning, unsupervised learning",
                 packages=["TreeOrdination"],
                 python_requires = ">=3.10",
                 install_requires = ["numpy == 1.23.3",
                                     "pandas == 1.5.0",
                                     "scikit-learn == 1.1.2",
                                     "scikit-bio == 0.5.7",
                                     "umap-learn == 0.5.3",
                                     "deicode == 0.2.4"
                                     ],
                 classifiers=["Programming Language :: Python :: 3.10+",
                              "License :: MIT License",
                              "Operating System :: OS Independent",
                              "Topic :: Ecology :: Biomarker Selection :: Metagenomics :: Supervised Learning :: Unsupervised Learning :: Metabarcoding :: Biomonitoring :: Ecological Assessment :: Machine Learning"],
                 )
