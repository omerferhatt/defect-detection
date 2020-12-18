import argparse
import os
from datetime import datetime

import tensorflow as tf

from data.dataset import get_ds_pipeline
from models.defect_model import DefectLocalizeModel
from utils.losses import NonZeroMSELoss, NonZeroL2Loss


def parse():
    parser = argparse.ArgumentParser(description='Defect detection benchmarking')
    # Data/Path arguments
    parser.add_argument('-d', '--raw-data', type=str, default='data/raw')
    parser.add_argument('-t', '--txt-data', type=str, default='data')
    parser.add_argument('-c', '--csv-data', type=str, default='data/csv')
    parser.add_argument('-ld', '--log-dir', type=str, default='logs')
    parser.add_argument('-L', '--load-model', type=str)
    # Training parameters
    parser.add_argument('-l', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('-e', '--epoch', type=int, default=20)
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-C', '--save-checkpoint', action='store_true')
    # Inference parameters
    parser.add_argument('-i', '--inference', action='store_true')
    parser.add_argument('-r', '--real-time')
    parser.add_argument('-T', '--test-image', nargs='?')
    parser.add_argument('-s', '--save-result', action='store_true')

    return parser.parse_args()


def main():
    if not arg.inference:
        train_ds, test_ds = get_ds_pipeline(os.path.join(arg.csv_data, 'train.csv'),
                                            os.path.join(arg.csv_data, 'test.csv'),
                                            batch_size=arg.batch_size)
        defect_cls = DefectLocalizeModel(backbone=tf.keras.applications.DenseNet169)
        defect_cls.model.summary()
        defect_cls.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=arg.learning_rate),
            loss={
                'is_def': tf.keras.losses.BinaryCrossentropy(),
                'cls': tf.keras.losses.CategoricalCrossentropy(),
                'bbox_param': NonZeroMSELoss(),
                'bbox_center': NonZeroL2Loss()},
            metrics={
                'is_def': tf.keras.metrics.BinaryAccuracy(),
                'cls': tf.keras.metrics.CategoricalAccuracy()},
            run_eagerly=True
        )

        if not os.path.exists(arg.log_dir):
            os.mkdir(arg.log_dir)
        now = datetime.now()
        date_time = now.strftime("%H_%M_%m_%d")

        callbacks = [tf.keras.callbacks.EarlyStopping(patience=10),
                     tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.4, verbose=1)]

        if arg.save_checkpoint:
            checkpoint_dir = os.path.join(arg.log_dir, date_time, 'checkpoint')
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(checkpoint_dir, verbose=1, save_best_only=True)
            )
            tensorboard_dir = os.path.join(arg.log_dir, date_time, 'tensorboard')
            callbacks.append(
                tf.keras.callbacks.TensorBoard(tensorboard_dir)
            )

        if arg.load_model is not None:
            defect_cls.model.load_weights(arg.load_model)

        defect_cls.model.fit(
            train_ds,
            epochs=arg.epoch,
            validation_data=test_ds,
            callbacks=callbacks,
            verbose=1
        )

    else:
        pass


if __name__ == '__main__':
    arg = parse()
    main()
