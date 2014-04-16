import numpy as np
import scipy.ndimage


def ring_kernel(params, sh):
    """Compute a gaussian ring kernel image
    Args:
        params: radius and standard deviation of the gaussian
        sh: x and y dimensions of the kernel
    """
    x = np.arange(sh[1])-sh[1]/2
    y = np.arange(sh[0])-sh[0]/2
    r = np.sqrt(y[:, np.newaxis]**2+x[np.newaxis, :]**2)

    r0, s = params
    kernel = np.exp(-(r-r0)**2/(2*s**2))
    kernel /= kernel.sum()
    return kernel


def chog_imgs(img):
    img = img.astype('double')
    param_sets = [[0, 2], [2, 2], [4, 2], [6, 2]]
    eps = 1e-9
    gamma = 0.8
    L = 5

    #w_func={'circle',[4,2]};

    l2 = True

    complex_derivatives = np.array([[0, 0, -1j/8, 0, 0], [0, 0, 1j, 0, 0],
                                    [-1j/8, 1, 0, -1, 1/8], [0, 0, -1j, 0, 0],
                                    [0, 0, 1j/8, 0, 0]], dtype='complex')

    # computing the complex derivatives df/dx+idf/dy (paper text below eq. (4))
    gradient_image = scipy.ndimage.correlate(img, np.real(complex_derivatives), mode='wrap')\
        + 1j*scipy.ndimage.correlate(img, np.imag(complex_derivatives), mode='wrap')

    # computing the gradient magnitude
    gradient_magnitude = np.abs(gradient_image)
    inv_gradient_magnitude = 1./(gradient_magnitude+eps)

    # gamma correction (paper eq. (4))
    gradient_magnitude = gradient_magnitude**gamma

    # computing gradient orientation ^g (paper text below eq. (1))
    gradient_direction = gradient_image*inv_gradient_magnitude

    Fourier_coefficients = np.zeros((L,)+np.shape(img), dtype='complex')

    Fourier_coefficients[0, :, :] = gradient_magnitude
    Fourier_coefficients[1, :, :] = gradient_direction*gradient_magnitude

    current = gradient_direction
    for l in range(2, L):
        current = current*gradient_direction
        Fourier_coefficients[l, :, :] = current*gradient_magnitude

    output = np.zeros((len(param_sets),)+np.shape(Fourier_coefficients), dtype='complex')

    for i, params in enumerate(param_sets):
        wf = ring_kernel(params, np.shape(img))
        tmp = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(Fourier_coefficients)*np.fft.fft2(wf)))
        output[i, :, :, :] = tmp/(abs(tmp)+1e-9)

    return output


def histfeat(chog_imgs, x, y, R=16):
    nw, nd, h, w = np.shape(chog_imgs)

    output = chog_imgs[:, :, y-R:y+R, x-R:x+R]
    features = np.array([np.sum((np.floor((np.real(output)+1)*32) == i), axis=(2, 3)) for i in range(64)]).flatten()
    return features


class CircularHistogramOfGradients(object):
    """Computes CHOG features"""
    def __init__(self, S):
        """Args:
        """
        super(CircularHistogramOfGradients, self).__init__()
        self.S = S
        self.image = None

    def feature_vector(self, image, x, y):
        """Returns the feature vector of an image for a patch centered at x, y.
        The first call requires convolution with multiple kernels,
         subsequent calls for the same image will be much faster.
        """
        if self.image is not image:
            self.image = image
            self.chog_imgs = chog_imgs(image)

        h, w = image.shape
        if 0 <= x-self.S/2. < w-self.S and 0 <= y-self.S/2. < h-self.S:
            return histfeat(self.chog_imgs, x, y, self.S/2.)

        else:
            raise ValueError("Requested patch exceeds boundaries of image.")
