import numpy as np


def normalize(x, epsilon=1e-9):
    """Normalizes the array x to zero mean and unit variance"""
    return (x-x.mean())/(x.std()+epsilon)


def whiteningMatrixZCA(X, epsilon=0.1):
    """Calculate the ZCA whitening matrix for a dataset
        Args:
            X: the dataset from which the whitening matrix should be calculated.
               Should be an MxN array where N is the number of samples and M is the vector length.
            epsilon: regularization parameter
        Returns:
            the whitening matrix W such that X_ZCA = WX
    """
    E, D, _ = np.linalg.svd(np.cov(X))
    W = E.dot(np.diag(1./(np.sqrt(D)+epsilon)).dot(E.T))
    return W


def randomPatch(images, S):
    """Select a patch at random from a collection of images
        Args:
            images: a list or array of images
            S: the size of the image patch to return
        Returns:
            an array of size SxS chosen at random from the images
    """

    if type(images) is list:
        ndim = 3
    elif type(images) is np.ndarray:
        ndim = len(images.shape)

    if ndim not in (1, 2, 3):
        raise ValueError("Can't understand shape of images")

    if ndim == 2:
        image = images
    else:
        idx = np.random.randint(len(images))
        image = images[idx]

    w, h = image.shape
    x = np.random.randint(0, w-S)
    y = np.random.randint(0, h-S)
    return image[y:y+S, x:x+S]


class RandomPatchSoftThreshold(object):
    """Computes features by soft thresholding of a dictionary of patches
     sampled at random from the provided images. Follows the method described
     in "The Importance of Encoding Versus Training with Sparse Coding and
     Vector Quantization", Coates and Ng 2011."""
    def __init__(self, images, S=16, N=1000, alpha=0.1):
        """Args:
                images: a list or array of images
                S: the size of the image patch
                N: the number of patches in the dictionary
                alpha: the threshold
        """
        super(RandomPatchSoftThreshold, self).__init__()
        self.S = S
        self.alpha = alpha

        self.D = np.zeros((S*S, N))
        for i in xrange(N):
            self.D[:, i] = normalize(randomPatch(images, S)).flatten()

        self.W = whiteningMatrixZCA(self.D)
        self.D = self.W.dot(self.D)

    def feature_vector(self, image, x, y):
        """Returns the feature vector of an image for a patch centered at x, y
        """
        h, w = image.shape
        if 0 <= x-self.S/2. < w-self.S and 0 <= y-self.S/2. < h-self.S:
            patch = image[y-self.S/2:y+self.S/2., x-self.S/2:x+self.S/2.]
            patch = patch-patch.mean()
            x = self.W.dot(patch.flatten())

            fplus = x.dot(self.D)-self.alpha
            fplus = fplus*(fplus > 0)
            fminus = -x.dot(self.D)-self.alpha
            fminus = fminus*(fminus > 0)

            return np.concatenate((fplus, fminus))
        else:
            raise ValueError("Requested patch exceeds boundaries of image.")
