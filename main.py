import tensorflow as tf

from data.dataset import get_ds_pipeline
from models.defect_model import DefectLocalizeModel
from utils.losses import NonZeroMSELoss, NonZeroL2Loss

if __name__ == '__main__':
    train_ds, test_ds = get_ds_pipeline(batch_size=64)
    defect_cls = DefectLocalizeModel(backbone=tf.keras.applications.DenseNet169)
    defect_cls.model.summary()
    defect_cls.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
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

    defect_cls.model.fit(train_ds, epochs=100, validation_data=test_ds,
                         callbacks=[tf.keras.callbacks.EarlyStopping(patience=11),
                                    tf.keras.callbacks.ReduceLROnPlateau(patience=6, verbose=1)])
