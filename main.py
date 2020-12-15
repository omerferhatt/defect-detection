import tensorflow as tf

from data.dataset import get_ds_pipeline
from models.defect_model import DefectLocalizeModel
from utils.losses import NonZeroEllipticBBoxLoss

if __name__ == '__main__':
    train_ds, test_ds = get_ds_pipeline()
    defect_cls = DefectLocalizeModel(backbone=tf.keras.applications.efficientnet.EfficientNetB0)
    # trainer = ModelTrainer(defect_cls.model, epoch=10)
    # trainer.set_optimizer(tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9, nesterov=True))
    # trainer.set_loss(NonZeroEllipticBBoxLoss())
    # trainer.set_metrics(tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.BinaryAccuracy())
    # trainer.training_loop(train_ds, test_ds)
    defect_cls.model.compile(
        optimizer='rmsprop',
        loss={
            'is_def': tf.keras.losses.BinaryCrossentropy(),
            'cls': tf.keras.losses.CategoricalCrossentropy(),
            'bbox': NonZeroEllipticBBoxLoss()},
        loss_weights=[1, 1, 2],
        metrics={
            'is_def': tf.keras.metrics.BinaryAccuracy(),
            'cls': tf.keras.metrics.CategoricalAccuracy()},
        run_eagerly=True
    )
    defect_cls.model.fit(train_ds, epochs=50, validation_data=test_ds)
