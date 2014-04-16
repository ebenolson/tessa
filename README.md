# tessa
A collection of Python modules for calculation of various texture features.

## Features

Tessa can calculates features using these methods:

* **SFTA** (Segmentation-based Fractal Texture Analysis)
* **CHOG** (Circular Histogram of Gradients)
* **RPST** (Random Patch Dictionary with Soft Thresholding)
* **TODO** ~~LPQ (Local Phase Quantization)~~
* **TODO** ~~RZM (Regional Zernike Moments)~~
* **TODO** ~~BGP (Binary Gabor Patterns)~~

## Requirements

Tessa is written and tested for Python 2.7, and requires numpy >= 1.8 and scipy >= 0.13.3. The examples also depend on scikit-learn for classification.

## Usage

**TODO**

## Acknowledgments and citations

Christoph Gohlke's excellent [tifffile](http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html) code is included for reading and writing image data.

A subset of the [Kylberg Texture Dataset](http://www.cb.uu.se/~gustaf/texture/) is include for testing and demonstration purposes.

The SFTA module implements the algorithm presented by Alceu Costa in *"An Efficient Algorithm for Fractal Analysis of Textures"*, SIBGRAPI 2012. It is based on his provided [MATLAB source code](http://www.mathworks.com/matlabcentral/fileexchange/37933-sfta-texture-extractor/content/sfta/sfta.m)

The CHOG module implements the algorithm published by Henrik Skibbe and Marco Reisert in *"
Circular Fourier-HOG features for rotation invariant object detection in biomedical images", ISBI 2012. It is based on their published [MATLAB source code](https://bitbucket.org/skibbe/fourierchog/wiki/Home)

The RPST module implements methods described by Adam Coates and Andrew Ng in *"The Importance of Encoding Versus Training with Sparse Coding and Vector Quantization"*, Proc. ICML 2011