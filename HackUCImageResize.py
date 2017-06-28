
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import cv2
import math
from glob import glob
import os
#lots of unnecessary imports


# In[38]:


#SINGLE IMAGE TEST
#originalimage = cv2.imread('pic/1.jpg')

#image = cv2.cvtColor(originalimage, cv2.COLOR_BGR2GRAY)

#r = 28.0 / image.shape[1]
#dim = (28, int(image.shape[0] * r))

# perform the actual resizing of the image and show it
#resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

#cropped = resized[0:256, 0:256]
#cv2.imshow("cropped", cropped)
#cv2.waitKey(0)

#print(cropped)
# write the cropped image to disk in PNG format
#cv2.imwrite("thumbnail.jpg", cropped)


# In[4]:


def to28(file_name):
    
    print('pic2/' + str(file_name) + '.jpg')
    
    originalimage = cv2.imread('pic2/' + str(file_name) + '.jpg')

    #image = cv2.cvtColor(originalimage, cv2.COLOR_BGR2GRAY)
    image = originalimage
    r = 64.0 / image.shape[1]
    dim = (64, int(image.shape[0] * r))

    # perform the actual resizing of the image and show it
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    cropped = resized[0:64, 0:64]
    
    #cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    if (cropped.shape[0] < 64):
        pix = 64 - cropped.shape[0]
        cropped = cv2.copyMakeBorder(cropped,0,pix,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])

    print(cropped)
    # write the cropped image to disk in PNG format
    cv2.imwrite(str(file_name) + ".jpg", cropped)


# In[32]:



#Rename pictures USE ONLY ONCE
import os
path = 'C:/Users/zhizh/Desktop/Project/HackUC2/pic2'
files = os.listdir(path)
i = 1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, str(i)+'.jpg'))
    i = i+1


# In[5]:


#RESIZE IMAGES 
for i in range(11403):
    #print(i+1)
    to28(i+1)
    


# In[ ]:




