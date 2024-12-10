import cv2
import numpy as np
import matplotlib.pyplot as plt

def loader(path: str):
    """
    cv2 method to load our image
    """
    return cv2.imread(path,cv2.IMREAD_ANYCOLOR)

def size(image: cv2):
    """
    Returns the shape attribute of our cv2 image
    """
    return image.shape

def histogram(image: cv2, purpose: str):
    
    plt.hist(image, bins='auto')
    plt.title('Histogram for {}'.format(purpose))
    plt.show()
