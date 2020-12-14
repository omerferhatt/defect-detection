import pathlib

import tensorflow as tf


class Dataset:
    def __init__(self, train_csv_path: str, test_csv_path: str, csv_endline: str, batch_size: int):
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.csv_endline = csv_endline
        self.batch_size = batch_size
        self.autotune = tf.data.experimental.AUTOTUNE
        self.__record_defaults = [str(), int(), int(), int(), int(), float(), float(), float(), float(), float()]

    def configure_ds(self, ds: tf.data.Dataset, is_training: False) -> tf.data.Dataset:
        if is_training:
            return ds.batch(self.batch_size).cache().prefetch(self.autotune)
        else:
            return ds.batch(self.batch_size)

    def get_csv_ds(self, lines) -> tf.data.Dataset:
        features = tuple(tf.io.decode_csv(lines, record_defaults=self.record_defaults))
        path_tensor = features[0]
        classification_tensors = features[1:4]
        is_defect_tensor = features[4]
        bbox_tensors = features[5:]
        ds = tf.data.Dataset.from_tensor_slices((path_tensor, classification_tensors, is_defect_tensor, bbox_tensors))
        return ds

    def read_csv_as_text(self, path: str):
        text = pathlib.Path(path).read_text()
        lines = text.split(self.csv_endline)[:-1]
        return lines

    @staticmethod
    def decode_ds(path, class_labels, is_defect, bbox):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_png(img_bytes)
        img = tf.image.grayscale_to_rgb(img)
        # img = tf.image.convert_image_dtype(img, tf.float32)
        class_labels = tf.stack(class_labels, axis=-1)
        bbox = tf.stack(bbox, axis=-1)
        img = tf.image.resize(img, size=(224, 224))
        return img, (class_labels, is_defect, bbox)

    @property
    def record_defaults(self):
        return self.__record_defaults


def get_ds_pipeline(train_csv_path='data/csv/train.csv', test_csv_path='data/csv/test.csv', batch_size=32):
    data = Dataset(train_csv_path, test_csv_path, csv_endline='\n', batch_size=batch_size)
    train_csv_lines, test_csv_lines = list(map(data.read_csv_as_text, (data.train_csv_path, data.test_csv_path)))
    train_ds = data.get_csv_ds(train_csv_lines)
    test_ds = data.get_csv_ds(test_csv_lines)
    train_ds = train_ds.map(data.decode_ds, num_parallel_calls=data.autotune)
    test_ds = test_ds.map(data.decode_ds, num_parallel_calls=data.autotune)
    train_ds = data.configure_ds(train_ds, is_training=True)
    test_ds = data.configure_ds(test_ds, is_training=True)
    return train_ds, test_ds


if __name__ == '__main__':
    train_ds, test_ds = get_ds_pipeline()
