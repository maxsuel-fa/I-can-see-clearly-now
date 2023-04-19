import matplotlib.pyplot as plt

from PIL import Image, ImageChops, ImageOps
from torch import Tensor
from typing import Tuple

def custom_add_subplot(figure: plt.figure,
                       image: Image,
                       grid_rows: int,
                       grid_cols: int,
                       grid_position: int,
                       sub_title: str = None,
                       cmap = None) -> None:
    """ TODO """
    figure.add_subplot(grid_rows, grid_cols, grid_position)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.title(sub_title)

def plot_result(clean_img: Image,
                rainy_img: Image,
                derained_img: Image,
                plot_size: Tuple[float, float] = (20, 5),
                save_path: str = None) -> None:
    """ TODO """
    figure = plt.figure(figsize=plot_size, constrained_layout=True)

    grid_rows = 1
    grid_cols = 4

    custom_add_subplot(figure,
                       clean_img,
                       grid_rows,
                       grid_cols,
                       1)

    custom_add_subplot(figure,
                       rainy_img,
                       grid_rows,
                       grid_cols,
                       2)

    custom_add_subplot(figure,
                       derained_img,
                       grid_rows,
                       grid_cols,
                       3)

    #clean_img = ImageOps.grayscale(clean_img)
    #derained_img = ImageOps.grayscale(derained_img)

    difference = ImageChops.difference(clean_img, derained_img).convert('L')
    
    custom_add_subplot(figure,
                       difference,
                       grid_rows,
                       grid_cols,
                       4, cmap='gray')

    if save_path != None:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

