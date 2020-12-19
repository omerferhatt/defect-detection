import argparse
import os
import time
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from data.dataset import get_ds_pipeline
from inference import test_single_image
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
    parser.add_argument('-T', '--test-image', nargs='+')
    parser.add_argument('-s', '--save-result', action='store_true')

    return parser.parse_args()


def main():
    # Training phase
    if not arg.inference:
        # Get dataset pipeline
        train_ds, test_ds = get_ds_pipeline(os.path.join(arg.csv_data, 'train.csv'),
                                            os.path.join(arg.csv_data, 'test.csv'),
                                            batch_size=arg.batch_size)
        defect_cls = DefectLocalizeModel(backbone=tf.keras.applications.DenseNet169)
        # Print model
        # defect_cls.model.summary()
        # Create log dir
        if not os.path.exists(arg.log_dir):
            os.mkdir(arg.log_dir)
        # Get date-time for saving
        now = datetime.now()
        date_time = now.strftime("%H_%M_%m_%d")
        # Callbacks for training session
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=10),
                     tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.1, verbose=1, cooldown=3, min_lr=1e-5)]

        # If specified, checkpoint callback will be created
        if arg.save_checkpoint:
            checkpoint_dir = os.path.join(arg.log_dir, date_time, 'checkpoint.ckpt')
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    checkpoint_dir,
                    save_weights_only=True,
                    verbose=1,
                    save_best_only=True
                )
            )
            tensorboard_dir = os.path.join(arg.log_dir, date_time, 'tensorboard')
            callbacks.append(
                tf.keras.callbacks.TensorBoard(tensorboard_dir)
            )
        # If specified, loads checkpoint from disk
        if arg.load_model is not None:
            defect_cls.model.load_weights(arg.load_model)

        # Compile model and build graph
        # Eager execution is necessary for custom loss
        defect_cls.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=arg.learning_rate),
            loss={
                'is_def': tf.keras.losses.BinaryCrossentropy(label_smoothing=0.2),
                'cls': tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                'bbox_param': NonZeroMSELoss(),
                'bbox_center': NonZeroL2Loss()},
            metrics={
                'is_def': tf.keras.metrics.BinaryAccuracy(),
                'cls': tf.keras.metrics.CategoricalAccuracy()},
            run_eagerly=True
        )

        # Training
        defect_cls.model.fit(
            train_ds,
            epochs=arg.epoch,
            validation_data=test_ds,
            callbacks=callbacks,
            verbose=1
        )

    # Inference phase
    else:
        # Create model object
        t = time.time()
        print('Creating model!')
        defect_cls = DefectLocalizeModel(backbone=tf.keras.applications.DenseNet169)
        tf.keras.utils.plot_model(
            defect_cls.model,
            to_file="model.png",
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=192,
        )
        print(f'Model created {time.time() - t:.5f} second\n')
        if arg.load_model is not None:
            # Load weights
            t = time.time()
            print('Weight load started!')
            defect_cls.model.load_weights(arg.load_model).expect_partial()
            print(f'Weight load: {time.time() - t:.5f} second\n')
            # Test image on model
            test_single_image(defect_cls.model, arg.test_image[0], arg.save_result)


if __name__ == '__main__':
    arg = parse()
    main()
