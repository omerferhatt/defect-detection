import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib.patches import Ellipse

select_im = 23


def test_model(model, test_ds, select_im=1):
    for idx, (x, y) in enumerate(test_ds.unbatch().take(select_im).as_numpy_iterator()):
        if idx == select_im - 1:
            img = x
            class_type = y[3]
            is_defect = y[0]
            bbox_param = y[1]
            bbox_center = y[2]
            img_arr = np.array(img)[np.newaxis, :, :, :]
            y_pred = model.predict(img_arr)
            img_arr = (img_arr * 255).astype(np.uint8)
            class_type_pred = y_pred[3]
            is_defect_pred = y_pred[0]
            bbox_param_pred = np.squeeze(y_pred[1])
            bbox_center_pred = np.squeeze(y_pred[2])
            img_arr = Image.fromarray(img_arr[0])
            img_arr = img_arr.resize((512, 512))
            print(f'Is defected: \tGT vs Prediction | {is_defect} - {is_defect_pred.squeeze()}')
            print(f'Class type:  \tGT vs Prediction | {np.argmax(class_type)} - {np.argmax(class_type_pred.squeeze())}')
            print(f'Bbox center: \tGT vs Prediction | {bbox_center} - {bbox_center_pred}')
            print(f'Bbox params: \tGT vs Prediction | {bbox_param} - {bbox_param_pred}')
            print(f'Bbox center  \tL2 Distance      | {np.mean((bbox_center - bbox_center_pred) ** 2)}')
            # Plot configuration
            fig, ax = plt.subplots(figsize=(12, 12))
            plt.imshow(img_arr, cmap='gray')
            # Creating un-filled ellipse on image
            if is_defect > 0.5:
                e = Ellipse(xy=(bbox_center_pred * 512), width=bbox_param_pred[0] * 256,
                            height=bbox_param_pred[1] * 256,
                            angle=((bbox_param_pred[2] * 2 * np.pi - np.pi) * 180 / np.pi), edgecolor='b', lw=2,
                            facecolor='none')
                e.set_alpha(0.8)
                ax.add_artist(e)
                e_org = Ellipse(xy=(bbox_center * 512), width=bbox_param[0] * 256, height=bbox_param[1] * 256,
                                angle=((bbox_param[2] * 2 * np.pi - np.pi) * 180 / np.pi), edgecolor='r', lw=2,
                                facecolor='none')
                e_org.set_alpha(0.8)
                ax.add_artist(e_org)
            plt.show()


def test_single_image(model: tf.keras.Model, path: str, save_result: bool):
    img_pil = Image.open(path)
    img_pil_resized = img_pil.resize((224, 224))
    img = np.array(img_pil_resized, dtype=np.float32) / 255.
    img = img[np.newaxis, :, :, :]
    pred = model.predict(img)
    is_defect_pred = pred[0]
    class_type_pred = pred[3]
    bbox_param_pred = np.squeeze(pred[1])
    bbox_center_pred = np.squeeze(pred[2])
    print(f'Is defected: \tPrediction | {is_defect_pred.squeeze()}')
    print(f'Class type:  \tPrediction | {np.argmax(class_type_pred.squeeze())}')
    print(f'Bbox center: \tPrediction | {bbox_center_pred}')
    print(f'Bbox params: \tPrediction | {bbox_param_pred}')
    img_original = np.array(img_pil, dtype=np.uint8)
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.imshow(img_original, cmap='gray')
    # Creating un-filled ellipse on image
    if is_defect_pred > 0.5:
        e = Ellipse(xy=(bbox_center_pred * 512), width=bbox_param_pred[0] * 256,
                    height=bbox_param_pred[1] * 256,
                    angle=((bbox_param_pred[2] * 2 * np.pi - np.pi) * 180 / np.pi), edgecolor='b', lw=2,
                    facecolor='none')
        e.set_alpha(0.8)
        ax.add_artist(e)
        plt.show()
        if save_result:
            fig.savefig('result.png')
