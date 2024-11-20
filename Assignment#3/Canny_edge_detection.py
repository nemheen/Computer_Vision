#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import math
import numpy as np


# In[196]:


#from HW2
def gauss1d(sigma):
    
    #sigma value must be positive value
    assert (sigma > 0), 'Sigma value should be positive'
    
    #length of the filter
    l = int(np.ceil(sigma * 6))
    
    #if l is an even, adding 1 to resulting in next odd int
    if l % 2 == 0:
        l += 1
    
    #calculating middle point where x will be calculated as distance value from the mid
    mid = l // 2
    
    #numpy.arange([start, ]stop, [step, ]dtype=None, *, like=None)
    x = np.arange(-mid, mid + 1) # +1 because start point is inclusive and end point is exclusive
    #x is now an array with 'l' length with values ranging from -mid to mid+1
    
    #each value of arr is computed using Gaussian function with e^(-(x**2)/(2 * sigma**2)) x being each value
    arr = np.exp(-(x**2)/(2 * sigma**2))
    
    #normalizing the array to be all the values sum up to be 1
    arr = arr / np.sum(arr)
    
    return arr


# In[197]:


#from HW2
def gauss2d(sigma):
    
    #1d gaussian array filter using gauss1d()
    temp = gauss1d(sigma)
    
    #numpy.outer(a, b, out=None)[source]
    #using np.outer to add a new axis to the existing array returned by gauss1d() method
    arr = np.outer(temp, temp)
    
    #normalizing the values in the filter so they sum to 1.
    arr = arr / np.sum(arr)
    #print(f'filt shape: ', arr.shape)
    
    return arr


# In[198]:


#from HW2

def convolve2d(array, filter):
    m, n = array.shape
    
    f = filter.shape[0] #since filter is square size
    
    #pad_m, pad_n = (fm - 1) // 2, (fn - 1) // 2  
    pad_size = (f - 1) // 2 #since padding height and width is equal
    
    #np.pad(array, pad_width, mode='constant', **kwargs)[source]
    #using np.pad to pad the input array image
    base = np.pad(array, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')
    #below comment is for my own understanding!
    #(pad_m, pad_m) represents the amount of padding to add above and below each values, adding more rows
    #(pad_n, pad_n) represents the amount of padding to add before and after each column values, adding more columns
    
    
    #creating np.arrays filled with 0s with same size as array using np.zeros()
    res = np.zeros_like(array)
    
    # computing the convolution using two loops as instructed
    for i in range(m):
        for j in range(n):
            cut = base[i:i+f, j:j+f] #the neighborhood area where filter covers
            res[i, j] = np.sum(cut * filter) #element-wise multiplication to compute the center element of the resulting array
            
    return res.astype(np.float32)


# In[199]:


#from HW2
def gaussconvolve2d(array, sigma):
    #generating a filter with my ‘gauss2d’
    filt = gauss2d(sigma)
    
    #then applying it to the array with ‘convolve2d(array, filter)’ to have gaussian 2d array
    return convolve2d(array, filt)


# In[200]:


#1. Noise Reduction

def reduce_noise(img):
    """ Return the gray scale gaussian filtered image with sigma=1.6
    Args:
        img: RGB image. Numpy array of shape (H, W, 3).
    Returns:
        res: gray scale gaussian filtered image (H, W).
    """
    #implement
    
    #switching to the grayscale image.
    img_grey = img.convert('L')
    
    #converting img_grey to array 
    img_arr = np.asarray(img_grey, dtype=np.float32)
    
    #blurring the image using gaussconvolve2d
    res = gaussconvolve2d(img_arr, sigma=1.6)
    
    return res


# In[201]:


#2. Finding the intensity gradient of the image

def sobel_filters(img):
    """ Returns gradient magnitude and direction of input img.
    Args:
        img: Grayscale image. Numpy array of shape (H, W).
    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction of gradient at each pixel in img.
            Numpy array of shape (H, W).
    Hints:
        - Use np.hypot and np.arctan2 to calculate square root and arctan
    """
    #implement 
    
    #setting X,Y Sobel filters according to instruction
    x_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32)
    y_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    
    #applying the convolve2d to obtain the intesity of x, y direction using sobel filters
    grad_x = convolve2d(img, x_filter)
    grad_y = convolve2d(img, y_filter)
    

    #equivalent to sqrt(grad_x**2 + grad_y**2)
    G = np.hypot(grad_x, grad_y)
    
    #equivalent to arctan(grad_y / grad_x)
    theta = np.arctan2(grad_y, grad_x)
    
    #mapping gradient values to values between 0-255.
    G = (255 * G / np.max(G)).astype('uint8')

    return (G, theta)


# In[202]:


#3. Non-Maximum Suppression

#The principle is: the algorithm goes through all the points on the gradient intensity 
#matrix and finds the pixels with the maximum value in the edge directions.

def non_max_suppression(G, theta):
    """ Performs non-maximum suppression.
    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).
    Returns:
        res: non-maxima suppressed image.
    """
    # apply Non-max-suppression according to the angle of the edge (0,45,90,135).
    h, w = G.shape
    
    res = np.copy(G)
    
    #converting to degree
    theta_deg = np.rad2deg(theta)
    #print(theta_deg)
    
    #doing it except for the edge part
    for i in range(1, h-1):
        for j in range(1, w-1):
            
            angle = theta_deg[i, j] % 180  #convert angle to be in the range [0, 180)
            
            try:
                
                #below a few lines of comments are for my own understanding!
                #If one those two pixels are more intense than the one being processed, 
                #then only the more intense one is kept.
                #f there are no pixels in the edge direction having more intense values, 
                #then the value of the current pixel is kept.

                # representing angle 0
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    if (G[i, j] <= G[i, j - 1]) or (G[i, j] <= G[i, j + 1]):
                        res[i, j] = 0


                # representing angle 45 
                elif 22.5 <= angle < 67.5 :
                    if (G[i, j] <= G[i - 1, j + 1]) or (G[i, j] <= G[i + 1, j - 1]):
                        res[i, j] = 0


                # representing angle 90 
                elif (67.5 <= angle < 112.5):
                    if (G[i, j] <= G[i - 1, j]) or (G[i, j] <= G[i + 1, j]):
                        res[i, j] = 0


                # Diagonal edge representing 135 angle
                elif (112.5 <= angle < 157.5):
                    if (G[i, j] <= G[i - 1, j - 1]) or (G[i, j] <= G[i + 1, j + 1]):
                        res[i, j] = 0
                    
            except IndexError as e:
                pass
    
    return res



# In[203]:


def double_thresholding(img):
    """ 
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: double_thresholded image.
    """
    #implement 
    
    #using the given equation to determine the high and low threshold values
    diff = img.max() - img.min()
    hthresh = img.min() + diff * 0.15
    lthresh = img.min() + diff * 0.03
    
    # Initialize the result image
    res = np.zeros_like(img, dtype=np.uint8)
    
    # Define intensity values for weak and strong edges
    weak = 80
    strong = 255
    
    # Identify strong, weak, and non-relevant pixels based on threshold values
    strong_i, strong_j = np.where(img > hthresh)
    weak_i, weak_j = np.where((img <= hthresh) & (img >= lthresh))
    
    # Map pixel values accordingly
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    print(res)
    return res



# In[204]:


def dfs(img, res, i, j, visited=[]):
    # calling dfs on (i, j) coordinate imply that
    #   1. the (i, j) is strong edge
    #   2. the (i, j) is weak edge connected to a strong edge
    # In case 2, it meets the condition to be a strong edge
    # therefore, change the value of the (i, j) which is weak edge to 255 which is strong edge
    res[i, j] = 255

    # mark the visitation
    visited.append((i, j))

    # examine (i, j)'s 8 neighbors
    # call dfs recursively if there is a weak edge
    for ii in range(i-1, i+2) :
        for jj in range(j-1, j+2) :
            if (img[ii, jj] == 80) and ((ii, jj) not in visited) :
                dfs(img, res, ii, jj, visited)


# In[205]:


def hysteresis(img):
    """ Find weak edges connected to strong edges and link them.
    Iterate over each pixel in strong_edges and perform depth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: hysteresised image.
    """
    #implement 
    
    H, W = img.shape
    
    #creating copy of img
    res = img.copy()
    
    #finding strong edge pixels
    strong_edges = np.argwhere(res == 255)
   
    # Apply DFS on each strong edge pixel to find connected weak edges
    for i, j in strong_edges:
        #initialize visited list for DFS
        visited = []
        
        if 0 < i < H - 1 and 0 < j < W - 1:
            dfs(img, res, i, j, visited)
    
    #setting weak pixel values to 0
    res[res==80] = 0
    
    return res


# In[206]:


def main():
    RGB_img = Image.open('iguana.bmp')

    noise_reduced_img = reduce_noise(RGB_img)
    Image.fromarray(noise_reduced_img.astype('uint8')).save('iguana_blurred.bmp', 'BMP')
    
    g, theta = sobel_filters(noise_reduced_img)
    Image.fromarray(g.astype('uint8')).save('iguana_sobel_gradient.bmp', 'BMP')
    Image.fromarray(theta.astype('uint8')).save('iguana_sobel_theta.bmp', 'BMP')

    non_max_suppression_img = non_max_suppression(g, theta)
    Image.fromarray(non_max_suppression_img.astype('uint8')).save('iguana_non_max_suppression.bmp', 'BMP')

    double_threshold_img = double_thresholding(non_max_suppression_img)
    Image.fromarray(double_threshold_img.astype('uint8')).save('iguana_double_thresholding.bmp', 'BMP')

    hysteresis_img = hysteresis(double_threshold_img)
    Image.fromarray(hysteresis_img.astype('uint8')).save('iguana_hysteresis.bmp', 'BMP')
    


# In[207]:


main()

