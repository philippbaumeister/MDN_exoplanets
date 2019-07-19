# Inferring interior structures of exoplanets with Mixture Density Networks

Trained machine learning models and python notebooks for [Paper].

### Required packages

- ``python >= 3.6.8``
- ``keras >= 2.1.6``
- ``numpy >= 1.15.0``
- ``scipy >= 1.2.0``
- ``matplotlib >= 3.0.2``
- ``tensorflow >= 1.12.0``
- ``tensorflow-probability >= 0.5.0``
- ``ipywidgets >= 7.4.2``
- ``joblib >= 0.13.2``

### How to use

* *MDN_exoplanets.ipynb* contains all the code to load the trained MDN models and predict the distribution of possible interior structures of a planet.
* The *mdn* directory contains the MDN layer code adopted from <https://github.com/cpmpercussion/keras-mdn-layer>.
* The *models* directory contains data scalers and the MDN models trained either with mass and radius of the planet, or with mass, radius, and k<sub>2</sub>.
