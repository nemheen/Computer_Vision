#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This is Assignment-2 to introduce the Gaussian filter for Computer Vision
#with two parts


# In[247]:


from PIL import Image
import numpy as np
import math


# In[2]:


#Part 1:Gaussian Filtering


# In[248]:


#ex1
def boxfilter(n):
    
    #signaling an error with an assert statement of n is not odd
    assert n % 2 != 0, 'Dimension must be odd'
    
    #creating NumPy array according to the instruction
    arr = np.ones((n, n), dtype=np.float32) / (n * n) 
    #python docs specifies the data type of the elements in the numpy array to be 32-bit floating-point numbers.
    
    return arr


# In[249]:


boxfilter(4)


# In[250]:


boxfilter(5)


# In[251]:


boxfilter(3)


# In[252]:


boxfilter(4)


# In[253]:


boxfilter(7)


# In[256]:


#ex2

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


# In[257]:


print(gauss1d(0.3))


# In[258]:


gauss1d(0.5)


# In[259]:


gauss1d(1)


# In[260]:


gauss1d(2)


# In[268]:


#ex3
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


# In[269]:


gauss2d(0.5)


# In[270]:


gauss2d(1)


# In[271]:


#ex4 (a)

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


# In[279]:


#ex4 (b)
def gaussconvolve2d(array, sigma):
    #generating a filter with my ‘gauss2d’
    filt = gauss2d(sigma)
    
    #then applying it to the array with ‘convolve2d(array, filter)’ to have gaussian 2d array
    return convolve2d(array, filt)


# In[281]:


#ex4 (c)
#did not find the image of dog in images.zip as instructored in the assignment material
#instead used image of lion
img = Image.open('images/3a_lion.bmp')
print (img.size, img.mode, img.format)

#converting into greyscale
img_grey = img.convert('L')

#converting into np.array
img_arr = np.asarray(img_grey, dtype=np.float32)

#applying ‘gaussconvolve2d’ with a sigma of 3 on the image
convolved_img_arr = gaussconvolve2d(img_arr, 3)

convolved_img_arr.shape #proving to be 2d array


# In[290]:


#ex4 (d)

#using Image from PIL to show both the original and filtered images.
img.show()
img_grey.save('lion_grey.png')

 #converting the array back to unsigned integer format
img_conv = Image.fromarray(convolved_img_arr.astype('uint8'))

img_conv.show()
img_conv.save('lion_conv.png')


# In[283]:


#Part 2: hybrid Images
sigma = 5 #sigma value to be used in the following exercise


# In[291]:


#ex1 low frequency
#with 1a image
img_1a = Image.open('images/1a_steve.bmp')

#checking the image size, if both images have same size, no need to resize
print(img_1a.size)

#splitting the channels
r, g, b = img_1a.split()

#each channels converted to np.array using np.asarray
r_arr = np.asarray(r, dtype=np.float32)
g_arr = np.asarray(g, dtype=np.float32)
b_arr = np.asarray(b, dtype=np.float32)


#using gaussconvolve2d to filter the channels of the image to blur
low_r_a = gaussconvolve2d(r_arr, sigma)
low_g_a = gaussconvolve2d(g_arr, sigma)
low_b_a = gaussconvolve2d(b_arr, sigma)

#forming in image back from each of the channels with unsigned integer format

low_r_img = Image.fromarray(low_r_a.astype('uint8'))

low_g_img = Image.fromarray(low_g_a.astype('uint8'))

low_b_img = Image.fromarray(low_b_a.astype('uint8'))

#merging the three channels to form the low frequency image version
low_img_1a = Image.merge('RGB', (low_r_img, low_g_img, low_b_img))

#displaying and saving the low frequency image version
low_img_1a.show()
low_img_1a.save('steve_low.png')


# In[292]:


#ex2 high frequency
#with 1b image

img_1b = Image.open('images/1b_mandela.bmp')

# img_1b = img_1b.resize(img_1a.size)
#checking the image size, if both images have same size, no need to resize
print(img_1b.size)

#splitting the channels
r, g, b = img_1b.split()

#each channels converted to np.array using np.asarray
r_arr = np.asarray(r, dtype=np.float32)
g_arr = np.asarray(g, dtype=np.float32)
b_arr = np.asarray(b, dtype=np.float32)

#using gaussconvolve2d to filter the channels of the image to blur
low_r = gaussconvolve2d(r_arr, sigma)
low_g = gaussconvolve2d(g_arr, sigma)
low_b = gaussconvolve2d(b_arr, sigma)

#forming in image back from each of the channels with unsigned integer format

low_r_img = Image.fromarray(low_r.astype('uint8'))

low_g_img = Image.fromarray(low_g.astype('uint8'))

low_b_img = Image.fromarray(low_b.astype('uint8'))

#irst computing a low frequency Gaussian filtered image
low_img_1b = Image.merge('RGB', (low_r_img, low_g_img, low_b_img))

# low_img_1b.show()
# low_img_1b.save('mandela_low.png')

# then subtracting it from the original per channels
high_r_b = r_arr - low_r
high_g_b = g_arr - low_g
high_b_b = b_arr - low_b

#visualized by adding 128 and converting to an unsigned integer format
high_r_norm = (high_r_b + 128).astype('uint8')
high_g_norm = (high_g_b + 128).astype('uint8')
high_b_norm = (high_b_b + 128).astype('uint8')


#forming the high frequency image per channels
high_r_img = Image.fromarray(high_r_norm)
high_g_img = Image.fromarray(high_g_norm)
high_b_img = Image.fromarray(high_b_norm)

#and merging all the channels to establish the high frequency version of the image
high_im_1b = Image.merge('RGB', (high_r_img, high_g_img, high_b_img))

#displaying and saving the low frequency image version
high_im_1b.show()
high_im_1b.save('mandela_high.png')



# In[293]:


#ex3

#adding the low and high frequency images (per channel)
combined_r = low_r_a + high_r_b
combined_g = low_g_a + high_g_b
combined_b = low_b_a + high_b_b

#clamping the values of pixels on the high and low end to ensure they are in the valid range (between 0 and 255) for the final image
combined_r = np.clip(combined_r, 0, 255)
combined_g = np.clip(combined_g, 0, 255)
combined_b = np.clip(combined_b, 0, 255)

#forming the image per channels
combined_r_img = Image.fromarray(combined_r.astype('uint8'))
combined_g_img = Image.fromarray(combined_g.astype('uint8'))
combined_b_img = Image.fromarray(combined_b.astype('uint8'))

#combining all the combined channels of images
combined_im = Image.merge('RGB', (combined_r_img, combined_g_img, combined_b_img))

#displaying and saving the hybrid image
combined_im.show()
combined_im.save('combined_1.png')


# In[ ]:




