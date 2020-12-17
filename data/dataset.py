import pathlib

import albumentations as A
import matplotlib.pyplot as plt
import tensorflow as tf


class Dataset:
    def __init__(self, train_csv_path: str, test_csv_path: str, csv_endline: str, batch_size: int):
        # Attributes
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.csv_endline = csv_endline
        self.batch_size = batch_size
        # Autotune for some unknown parameters for creating dataset
        self.autotune = tf.data.experimental.AUTOTUNE
        # Default *.csv column data type order
        self.__record_defaults = [str(), float(), float(), float(),
                                  float(), float(), float(), float(),
                                  float(), float()]
        # Augmentation pipeline
        self.transforms = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.RandomGamma(p=0.4),
            A.ImageCompression(quality_lower=80, quality_upper=100, p=0.5),
            A.Blur(blur_limit=3, p=0.2)
        ])

    def aug_func(self, image):
        data = {"image": image}
        aug_data = self.transforms(**data)
        aug_img = aug_data["image"]
        return aug_img

    def aug_process(self, image, label):
        aug_img = tf.numpy_function(func=self.aug_func, inp=[image], Tout=tf.float32)
        return aug_img, label

    def apply_augmentation(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        ds = ds.map(self.aug_process, num_parallel_calls=self.autotune)
        return ds

    def configure_ds(self, ds: tf.data.Dataset, is_training: False) -> tf.data.Dataset:
        """Only training dataset needs cache and prefetching. Both dataset batched."""
        if is_training:
            return ds.batch(self.batch_size).cache().prefetch(self.autotune)
        else:
            return ds.batch(self.batch_size)

    def get_csv_ds(self, lines) -> tf.data.Dataset:
        # Decodes separator
        features = tuple(tf.io.decode_csv(lines, record_defaults=self.record_defaults))
        # Reading related fields into variables
        path_tensor = features[0]
        classification_tensors = features[1:4]
        is_defect_tensor = features[4]
        bbox_tensors = features[5:]
        # Creates tensor object from data
        ds = tf.data.Dataset.from_tensor_slices((path_tensor, classification_tensors, is_defect_tensor, bbox_tensors))
        return ds

    def read_csv_as_text(self, path: str) -> list:
        """
        Reads csv file from path as normal string array
        :param path: *.csv file path to read
        :return: Newline separated text
        """
        text = pathlib.Path(path).read_text()
        lines = text.split(self.csv_endline)[:-1]
        return lines

    @staticmethod
    def decode_ds(path, class_labels, is_defect, bbox):
        """
        This function designed to read single csv data row. It'll be parallelized with tf.data.Dataset.map()
        :param path: Image file path to read
        :param class_labels: One hot encoded class labels
        :param is_defect: Does image has and kind of defect.
        :param bbox: Elliptical bbox values
        :return: (image, (is_defect, class_labels, bbox_param, bbox_center)) type tensor
        """
        # Reads file as binary format from disk
        img_bytes = tf.io.read_file(path)
        # Whole images have to same extension, this part only allow as to read same type images
        # tf.image.decode_image function can read image files but technically it's not returning tensor
        # so we cannot move further with it.
        img = tf.image.decode_png(img_bytes)
        # Converting operation needed to make image compatible with related state-of-the-art model input.
        # Adds new channel to last axis
        img = tf.image.grayscale_to_rgb(img)
        # Auto normalize between 0-255 to 0-1
        img = tf.image.convert_image_dtype(img, tf.float32)
        # Resizing image before loading to GPU
        img = tf.image.resize(img, size=(224, 224))
        # Creates class label tensor from one-hot encoded features
        class_labels = tf.stack(class_labels, axis=-1)
        # Creates bbox tensor from parametric and center values
        bbox = tf.stack(bbox, axis=-1)
        # Return type is:
        # Resized 1 channel image,
        # Is image defected (Binary), Image class (Multi-Class),
        # Elliptic BBox parametric values (Semi-axes, Rotation angle), Elliptic BBox center values (X-center, Y-center)
        return img, (is_defect, class_labels, bbox[:3], bbox[3:])

    @property
    def record_defaults(self):
        """Default *.csv column data type order"""
        return self.__record_defaults


def get_ds_pipeline(train_csv_path='data/csv/train.csv', test_csv_path='data/csv/test.csv', batch_size=32) -> tuple:
    """
    Creates tf.data.Dataset object from train and test *.csv data. Process dataset and makes batches
    :param train_csv_path: Train *.csv file path
    :param test_csv_path: Test *.csv file path
    :param batch_size: Batch size for dataset
    :return: tf.data.Dataset object for both train and test datasets
    """
    # Creates dataset class to read csv files
    data = Dataset(train_csv_path, test_csv_path, csv_endline='\n', batch_size=batch_size)
    # Reads csv files line-by-line with seperetor
    train_csv_lines, test_csv_lines = list(map(data.read_csv_as_text, (data.train_csv_path, data.test_csv_path)))
    # Reads separated values into dataset and converts to tensor
    tr_ds = data.get_csv_ds(train_csv_lines)
    te_ds = data.get_csv_ds(test_csv_lines)
    # Converts paths to images and concatenate all labels
    tr_ds = tr_ds.map(data.decode_ds, num_parallel_calls=data.autotune)
    te_ds = te_ds.map(data.decode_ds, num_parallel_calls=data.autotune)
    # Data augmentation on training dataset
    tr_ds = data.apply_augmentation(tr_ds)
    # Enables some cache and pre-fetch features
    tr_ds = data.configure_ds(tr_ds, is_training=True)
    te_ds = data.configure_ds(te_ds, is_training=True)
    return tr_ds, te_ds


def visualize_images(image_list: list):
    fig, axs = plt.subplots(ncols=len(image_list) // 2, nrows=2, figsize=(15, 10), dpi=400)
    for idx, im in enumerate(image_list):
        r = idx // 5
        c = idx % 5
        im = (im * 255).astype(int)
        axs[r, c].imshow(im)
    plt.show()


if __name__ == '__main__':
    train_ds, test_ds = get_ds_pipeline()
    img_list = []
    for x, y in train_ds.unbatch().batch(1).take(10).as_numpy_iterator():
        img_list.append(x[0])

    visualize_images(img_list)
