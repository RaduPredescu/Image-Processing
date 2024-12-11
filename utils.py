import cv2
import numpy as np
import matplotlib.pyplot as plt


def loader(path: str):
    """
    cv2 method to load our image
    """
    return cv2.imread(path, cv2.IMREAD_ANYCOLOR)


def size(image: np.ndarray):
    """
    Returns the shape attribute of our cv2 image
    """
    return image.shape


def transform_to_gray(image: np.ndarray):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def add_gaussian_noise(gray_image: np.ndarray, gray_image_shape: tuple, config):
    sigma = config.zgomot.dispersie
    medie = config.zgomot.medie
    n = np.random.normal(medie, sigma, gray_image_shape)
    img_with_noise = gray_image + n
    img_with_noise = np.clip(img_with_noise, 0, 255).astype(np.uint8)
    return img_with_noise


def histogram(image: np.ndarray, purpose: str):
    if purpose == "color":
        colors = ("b", "g", "r")
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
            plt.xlim([0, 256])

        plt.title("Histograma pentru imaginea {}".format(purpose))
        plt.xlabel("Valoare pixel")
        plt.ylabel("Frecventa")
        plt.show()

    elif (
        purpose == "gri cu zgomot"
        or purpose == "gri"
        or purpose == "gri cu zgomot, filtrata"
    ):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])

        plt.figure()
        plt.title("Histograma pentru imaginea {}".format(purpose))
        plt.xlabel("Valoare pixel")
        plt.ylabel("Frecventa")
        plt.plot(hist, color="black")  # Plot in black for grayscale
        plt.xlim([0, 256])  # Pixel intensity range
        plt.grid()
        plt.show()


def apply_filter(img: np.ndarray, size: int):
    h, w = img.shape
    capat = size // 2
    # pregatire imagine noua cu zero
    new_img = np.zeros([h, w])
    POS = np.zeros([h, w])
    # parcurgerea imaginii ( atentie la capete )
    for i in range(capat, h - capat):
        for j in range(capat, w - capat):
            # extrage vecinatate
            vec = img[i - capat : i + capat + 1, j - capat : j + capat + 1]
            # populeaza imaginea noua cu rezultatul filtrarii
            new_img[i, j], POS[i, j] = adaptiv_orientat(vec, size)
    new_img = np.uint8(new_img)
    # intoarce noul rezultat
    return new_img, POS


def adaptiv_orientat(vec: np.ndarray, size: int):
    center = size // 2
    # medie orizontala ( 0 grade )
    med0 = np.sum(vec[center, :]) / size
    # medie verticala ( 90 grade )
    med90 = np.sum(vec[:, center]) / size
    # medie diagonala 45 grade
    med45 = 0.0
    for i in range(0, size):
        med45 = med45 + vec[i, size - 1 - i]
    med45 = med45 / size
    # medie diagonala 135 grade
    med135 = 0.0
    for i in range(0, size):
        med135 = med135 + vec[i, i]
    med135 = med135 / size
    # calculam diferentele
    # functia pentru modul ( valoare absoluta) este abs()
    temp = np.array([med0, med90, med45, med135])
    diff = np.abs(temp - vec[center, center])
    pos = np.argmin(diff)
    return temp[pos], pos


def mse(img: np.ndarray, img_modif: np.ndarray, capat: int):
    h, w = img.shape
    delta = (
        img[capat : h - capat, capat : w - capat]
        - img_modif[capat : h - capat, capat : w - capat]
    )
    delta = delta**2
    return np.sum(delta) / ((h - capat * 2) * (w - capat * 2))
