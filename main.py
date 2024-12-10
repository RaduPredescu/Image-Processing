from utils import loader,size,histogram
import matplotlib.pyplot as plt
import cv2
import numpy as np

if __name__ == '__main__':
    path = str(input('Enter path: '))
    image = loader(path)
    image_shape = size(image)
    
    #firstly, we want to save our image, to see if it is loaded properly    
    cv2.imwrite('saved_image.png',image)

    histogram(image, 'color')
    
