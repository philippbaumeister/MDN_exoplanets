# Inferring interior structures of exoplanets with Mixture Density Networks
![MIT License](https://img.shields.io/github/license/philippbaumeister/MDN_exoplanets.svg?style=flat-square) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.3556208-blue?style=flat-square)](https://zenodo.org/badge/latestdoi/188444287)

This repository contains the trained machine learning models and python notebooks for the paper [Machine-learning inference of the interior structure of low-mass exoplanets](https://doi.org/10.3847/1538-4357/ab5d32) (Baumeister et al. 2020).

### Required packages

This project requires Python 3.

- ``keras = 2.2.4``
- ``numpy = 1.18.0``
- ``scipy >= 1.2.0``
- ``matplotlib >= 3.0.2``
- ``tensorflow = 1.15.2``
- ``tensorflow-probability = 0.7.0``
- ``ipywidgets >= 7.4.2``
- ``joblib >= 0.13.2``
- ``scikit-learn = 0.22.1``

#### Installing the required packages

##### Using anaconda (preferred)
```
conda env create -f requirements.yml
```
Activate with
```
conda activate tf1.15
```

##### Using pip
```
pip install -r requirements.txt
```

### How to use

* *MDN_exoplanets.ipynb* contains all the code to load the trained MDN models and predict the distribution of possible interior structures of a planet.
* The *mdn* directory contains the MDN layer code adopted from <https://github.com/cpmpercussion/keras-mdn-layer>.
* The *models* directory contains data scalers and the MDN models trained either with mass and radius of the planet, or with mass, radius, and k<sub>2</sub>.
