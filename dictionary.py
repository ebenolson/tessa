import numpy as np


def whitenZCA():
    pass


def randomPatch(images, R):
    pass


class RandomPatchSoftThreshold(object):
    """Computes features by soft thresholding of a dictionary of patches
     sampled at random from the provided images. Follows the method described
     in "The Importance of Encoding Versus Training with Sparse Coding and
     Vector Quantization", Coates and Ng 2011."""
    def __init__(self, images, R=6, N=1000):
        super(RandomPatchSoftThreshold, self).__init__()
        self.D = np.array((whitenZCA(randomPatch(images, R)) for i in range(N)))

    def features(image, x, y, alpha=0.1):
        pass
