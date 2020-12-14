import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Ellipse


def test_model(model, test_ds, select_im=1):
    for idx, (x, y) in enumerate(test_ds.unbatch().take(select_im).as_numpy_iterator()):
        if idx == select_im - 1:
            img = x
            class_type = y[6:]
            is_defect = y[:1]
            bbox = y[1:6]
            print(class_type)
            print(is_defect)
            print(bbox)
            img_arr = np.array(img, dtype=np.uint8)
            y_pred = model.predict(img)
            bbox_pred = np.array(y_pred[1:6])
            img_arr = Image.fromarray(img_arr)
            img_arr = img_arr.resize((512, 512))
            # Plot configuration
            fig, ax = plt.subplots(figsize=(12, 12))
            plt.imshow(img_arr, cmap='gray')
            # Creating un-filled ellipse on image
            e = Ellipse(xy=(bbox_pred[3] * 512, bbox_pred[4] * 512), width=bbox_pred[0] * 256,
                        height=bbox_pred[1] * 256,
                        angle=((bbox_pred[2] * 2 * np.pi - np.pi) * 180 / np.pi), edgecolor='b', lw=2, facecolor='none')
            e_org = Ellipse(xy=(bbox[3] * 512, bbox[4] * 512), width=bbox[0] * 256, height=bbox[1] * 256,
                            angle=((bbox[2] * 2 * np.pi - np.pi) * 180 / np.pi), edgecolor='r', lw=2, facecolor='none')
            # Adding some transparency
            e.set_alpha(0.8)
            e_org.set_alpha(0.8)
            ax.add_artist(e)
            ax.add_artist(e_org)
            plt.show()
