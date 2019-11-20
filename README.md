# Inferring interior structures of exoplanets with Mixture Density Networks
![MIT License](https://img.shields.io/github/license/philippbaumeister/MDN_exoplanets.svg?style=flat-square)

This repository contains the trained machine learning models and python notebooks for the paper [Machine learning inference of the interior structure of low-mass exoplanets]() (Baumeister et al. 2019, in review).

### Required packages

This project requires Python 3.

- ``keras = 2.1.6``
- ``numpy = 1.15.0``
- ``scipy >= 1.2.0``
- ``matplotlib >= 3.0.2``
- ``tensorflow = 1.12.0``
- ``tensorflow-probability = 0.5.0``
- ``ipywidgets >= 7.4.2``
- ``joblib >= 0.13.2``
- ``scikit-learn = 0.21.3``

#### Installing the required packages

##### Using pip
```
pip install -r requirements.txt
```

##### Using anaconda
```
conda env create -f requirements.yml
```

### How to use

* *MDN_exoplanets.ipynb* contains all the code to load the trained MDN models and predict the distribution of possible interior structures of a planet.
* The *mdn* directory contains the MDN layer code adopted from <https://github.com/cpmpercussion/keras-mdn-layer>.
* The *models* directory contains data scalers and the MDN models trained either with mass and radius of the planet, or with mass, radius, and k<sub>2</sub>.
