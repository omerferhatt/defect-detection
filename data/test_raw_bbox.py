import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Ellipse


def show_defected_images(class_type=1, select_images=(0, 5)):
    """
    Shows selected defected images with elliptic bounding boxes.
    :param class_type: Selected class type,
    :param select_images: List or tuple which contains image number
    """
    assert class_type in [1, 2, 3]
    assert isinstance(select_images, (tuple, list))
    # Reads *.txt file to read raw data to plot image and bbox
    a_file = open(f'data/Class{class_type}_def.txt')
    for position, line in enumerate(a_file):
        # Selecting images from lines
        if position in select_images:
            # Getting image path
            test_image = f'data/raw/Class{class_type}_def/{position + 1:03}_{class_type}.png'
            # Reading image as binary and converting np.ndarray
            im = Image.open(test_image)
            im = np.asarray(im)
            # Tuple unpacking for all information to create ellipse
            # The last slicing is to get rid of the last newline character
            semi_major, semi_minor, rotation_angle, x_to_center, y_to_center = line[:-1].split('\t ')[1:]
            # Plot configuration
            a = plt.subplot(111, aspect='equal')
            plt.imshow(im, cmap='gray')
            # Creating un-filled ellipse on image
            e = Ellipse(xy=(float(x_to_center), float(y_to_center)), width=float(semi_major), height=float(semi_minor),
                        angle=(float(rotation_angle) * 180 / np.pi), edgecolor='b', lw=2, facecolor='none')
            # Adding some transparency
            e.set_alpha(0.8)
            # Adding patch to figure
            a.add_artist(e)
            plt.show()


if __name__ == '__main__':
    # 10th and 20th images of 2nd class will be showed
    show_defected_images(class_type=2, select_images=(10, 20))
