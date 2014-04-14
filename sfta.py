import numpy as np


def mrange(start, step, end):
    def gen(start, step, end):
        n = start
        while n <= end:
            yield n
            n = n+step
    return list(gen(start, step, end))


def findBorders(Im):
    I = np.pad(Im, [[1, 1], [1, 1]], 'constant', constant_values=1).astype('uint8')

    I2 = I[2:, 1:-1]+I[0:-2, 1:-1]+I[1:-1:, 2:]+I[1:-1:, 0:-2] + \
        I[2:, 2:]+I[2:, 0:-2]+I[0:-2, 2:]+I[0:-2, 0:-2]
    return Im * (I2 < 8)


def otsu(counts):
    p = counts*1.0/np.sum(counts)
    omega = np.cumsum(p)
    mu = np.cumsum(p*range(1, len(p)+1))
    mu_t = mu[-1]

    sigma_b_squared = (mu_t * omega - mu)**2 / (omega * (1-omega))
    maxval = np.max(np.nan_to_num(sigma_b_squared))
    if np.isnan(sigma_b_squared).all():
        pos = 0
    else:
        pos = mean((sigma_b_squared == maxval).nonzero())+1
    return pos


def otsurec(I, ttotal):
    if I == []:
        T = []
    else:
        I = I.astype(uint8).flatten()

        num_bins = 256
        counts = np.histogram(I, range(num_bins))[0]

        T = zeros((ttotal, 1))

        def otsurec_helper(lowerBin, upperBin, tLower, tUpper):
            if ((tUpper < tLower) or (lowerBin >= upperBin)):
                return
            level = otsu(counts[int(np.ceil(lowerBin))-1:int(np.ceil(upperBin))]) + lowerBin

            insertPos = np.ceil((tLower + tUpper) / 2.)
            T[insertPos-1] = level / num_bins
            otsurec_helper(lowerBin, level, tLower, insertPos - 1)
            otsurec_helper(level + 1, upperBin, insertPos + 1, tUpper)

        otsurec_helper(1, num_bins, 1, ttotal)
    return [t[0] for t in T]


def hausDim(I):
    maxDim = np.max(shape(I))
    newDimSize = 2**np.ceil(np.log2(maxDim))
    rowPad = newDimSize - shape(I)[0]
    colPad = newDimSize - shape(I)[1]

    I = np.pad(I, ((0, rowPad), (0, colPad)), 'constant')

    boxCounts = np.zeros(np.ceil(np.log2(maxDim))+1)
    resolutions = np.zeros(np.ceil(np.log2(maxDim))+1)

    iSize = shape(I)[0]
    boxSize = 1
    idx = 0
    while boxSize <= iSize:
        boxCount = sum(I > 0)
        idx = idx + 1
        boxCounts[idx-1] = boxCount
        resolutions[idx-1] = 1./boxSize

        boxSize = boxSize*2
        I = I[::2, ::2]+I[1::2, ::2]+I[1::2, 1::2]+I[::2, 1::2]
    D = np.polyfit(np.log(resolutions), np.log(boxCounts), 1)
    return D[0]


def sfta(I, nt):
    if len(shape(I)) == 3:
        I = mean(I, 2)
    elif len(shape(I)) != 2:
        raise ImageDimensionError

    I = I.astype(uint8)

    T = otsurec(I, nt)
    dSize = len(T)*6
    D = np.zeros(dSize)
    pos = 0
    for t in range(len(T)):
        thresh = T[t]
        Ib = I > (thresh*255)
        Ib = findBorders(Ib)

        vals = I[Ib.nonzero()].astype(double)
        D[pos] = hausDim(Ib)
        pos += 1

        D[pos] = mean(vals)
        pos += 1

        D[pos] = len(vals)
        pos += 1

    T = T+[1.0, ]
    for t in range(len(T)-1):
        lowerThresh = T[t]
        upperThresh = T[t+1]
        Ib = (I > (lowerThresh*255)) * (I < (upperThresh*255))
        Ib = findBorders(Ib)

        vals = I[Ib.nonzero()].astype(double)
        D[pos] = hausDim(Ib)
        pos += 1

        D[pos] = mean(vals)
        pos += 1

        D[pos] = len(vals)
        pos += 1
    return D


class SegmentationFractalTextureAnalysis(object):
    """Computes features by applying multiple thresholds and caculating the fractal dimension
        of the resulting binary images"""
    def __init__(self, nt):
        """Args:
                nt: the number of thresholds
        """
        super(SegmentationFractalTextureAnalysis, self).__init__()
        self.nt = nt

    def feature_vector(self, image):
        """Returns the feature vector of an image
        """
        return sfta(image, self.nt)
