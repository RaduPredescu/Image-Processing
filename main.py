from utils import (
    loader,
    size,
    transform_to_gray,
    histogram,
    add_gaussian_noise,
    apply_filter,
    mse,
)
from configuration import Config, read_params
import cv2


if __name__ == "__main__":
    """
    Imaginea incarcata trebuie sa fie neaparat RGB, altfel programul nu va merge :)
    """
    params = read_params("config.yaml")
    config = Config(params=params)

    path = str(input("Enter path: "))
    raw_image = loader(path)
    raw_image_shape = size(raw_image)

    # verificam daca am incarcat o imagine rgb si afisam dimensiunea
    assert len(raw_image_shape) == 3 and raw_image_shape[2] == 3, "Nu este rgb"
    print("Dimensiunea imaginii initiale este: {}".format(raw_image_shape))

    # salvam imaginea ca sa fim siguri ca am incarcat-o corect
    cv2.imwrite("loaded_image.png", raw_image)

    # afisam histograma pentru imaginea initiala
    histogram(raw_image, 'color')

    # transformam imaginea in gri
    gray_image = transform_to_gray(raw_image)

    # afisam dimensiunea imaginii gri
    # daca am procedat corect, ar trebui sa vedem lipsa unei dimensiuni sau valoarea 1 in loc de 3
    gray_image_shape = size(gray_image)
    print("Dimensiunea imaginii gri este: {}".format(gray_image_shape))
    cv2.imwrite("gray_image.png", gray_image)

    # afisam histograma pentru imaginea gri
    histogram(gray_image, 'gri')

    img_with_noise = add_gaussian_noise(gray_image, gray_image_shape, config)
    histogram(img_with_noise, 'gri cu zgomot')

    filtered_image, _ = apply_filter(img_with_noise, config.filtru.size)

    cv2.imwrite("filtered.png", filtered_image)
    histogram(filtered_image,'gri cu zgomot, filtrata')

    print("MSE pentru imaginea gri si imaginea cu zgomot: {}".format(mse(gray_image, img_with_noise, config.mse.capat)))
    print("MSE pentru imaginea gri si imaginea filtrata: {}".format(mse(gray_image, filtered_image, config.mse.capat)))
