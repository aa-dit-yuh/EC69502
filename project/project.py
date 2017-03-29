
# coding: utf-8

# # Multiscale Edge-based Text Extraction from Complex Images
# 
# From: [Xiaoqing Liu and Jagath Samarabandu; Multiscale Edge-based Text Region Extraction from Complex Images](https://pdfs.semanticscholar.org/b49b/3bfc48a6f9fa03889b219233f5fcc248e747.pdf), Proceedings of the IEEE International Conference on Mechatronics & Automation Niagara Falls, Canada, July 2005.

# # 1. Introduction
# 
# Text that appears in images contains important and useful information. Detection and extraction of text in images have been used in many applications. In this mini project, we will try to implement a multiscale edge-based text extraction algorithm, which can automatically detect and extract text in complex images. The proposed method is a general-purpose text detection and extraction algorithm, which can deal not only with printed document images but also with scene text. It is robust with respect to the font size, style, color, orientation, and alignment of text and can be used in a large variety of application fields, such as mobile robot navigation, vehicle license detection and recognition, object identification, document retrieving, page segmentation, etc.

# In[91]:

get_ipython().magic('matplotlib inline')

import numpy as np
from scipy import ndimage
from skimage import io, transform, measure, data, color, filters, morphology, exposure, img_as_bool, img_as_uint, img_as_float

images = io.imread_collection('examples/*')
image = images[2]
io.imshow(image)


# # 2. Methodology
# 
# ## 2.1 Candidate Text Detection
# 
# This stage aims to build a feature map by using three important properties of edges: edge strength, density and variance of orientations. The feature map is a gray-scale image with the same size of the input image, where the pixel intensity represents the possibility of text.
# 
# ### 2.1.1 Multi-scale edge detector
# 
# A convolution operation with a compass operator (as shown in Fig. 1) results in four oriented edge intensity images `E(θ),(θ ∈ {0,45,90,135})`, which contain all the properties of edges required in our proposed method.

# In[2]:

compass_kernels = {
    0: [
        [ -1, -1, -1],
        [  2,  2,  2],
        [ -1, -1, -1],
    ],
    45: [
        [ -1, -1,  2],
        [ -1,  2, -1],
        [  2, -1, -1],
    ],
    90: [
        [ -1,  2, -1],
        [ -1,  2, -1],
        [ -1,  2, -1],
    ],
    135:[
        [  2, -1, -1],
        [ -1,  2, -1],
        [ -1, -1,  2],
    ],
}


# In[96]:

image = color.rgb2gray(image)
feature_map = np.zeros(image.shape)

for octave in transform.pyramid_gaussian(image, max_layer=4, downscale=2):

    # 1: Directional Filtering
    edges = {}
    for orientation, kernel in compass_kernels.items():
        edges[orientation] = abs(ndimage.convolve(octave, kernel, mode='constant'))
        edges[orientation] /= edges[orientation].max()

    if not np.any(edges[90]):
        break

    # io.imshow(edges[90])

    # 2: Edge Selection
    # 2.1: Get strong edges
    e90_strong = edges[90] >= filters.threshold_otsu(edges[90])

    # io.imshow(e90_strong)

    # 2.2: Get weak edges
    ## 2.2 a): Dilate
    dilated = morphology.dilation(e90_strong)
    ## 2.2 b): Close
    closed  = morphology.closing(dilated)

    # io.imshow(closed)

    ## 2.2 c): E90w = |E90 x (closed-dilated)|z
    ##         where |.|z is the Otsu threshold operator
    e90_weak = edges[90] * abs(closed - dilated)
    e90_weak = e90_weak / abs(e90_weak).max()

    if not np.any(e90_weak):
        break

    e90_weak = e90_weak >= filters.threshold_mean(e90_weak)

    # io.imshow(e90_weak)

    e90 = e90_strong + e90_weak
    e90 = e90 >= filters.threshold_otsu(e90)

    # io.imshow(e90)

    # 3: Thin the edges
    e90_thin = morphology.thin(e90)

    # io.imshow(e90_thin)

    # 4: Label edges
    labels, num_labels = measure.label(e90_thin, return_num=True)
    e90_labelled = labels.copy()
    for cc in range(1, num_labels):
        e90_labelled[labels == cc] = np.count_nonzero(labels == cc)
    # e90_labelled = e90_labelled / e90_labelled.max()

    # io.imshow(e90_labelled)

    thresh = 0.2 * abs(e90_labelled).max()
    e90_short_thresh = np.zeros(e90_labelled.shape)
    e90_short_thresh[e90_labelled >= thresh] = 1

    # io.imshow(e90_short_thresh)

    e90_short = e90_labelled.copy()
    e90_short[e90_short_thresh == 1] = 0

    # io.imshow(e90_short)

    # 5: Feature Map Generation
    ## 5.1: Dilation
    e90_candidate = morphology.dilation(e90_short)

    # io.imshow(e90_candidate)

    ## 5.2: refined = candidate * sum(e0, e45, e90, e135)
    refined = np.multiply(e90_candidate, sum(edges.values()))
    refined = refined / abs(refined).max()

    # io.imshow(refined)

    ## 5.3:
    W = sum([filters.rank.maximum(edge, np.ones(25).reshape(5, 5)/25) for edge in edges.values()])/4
    fmap = np.multiply(refined, W)

    feature_map += transform.resize(fmap, image.shape)
# feature_map = fmap

feature_map = feature_map >= filters.threshold_otsu(feature_map)
io.imshow(feature_map)


# In[ ]:



