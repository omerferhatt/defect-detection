import tensorflow as tf

from data.dataset import get_ds_pipeline
from models.defect_model import DefectLocalizeModel
from train import ModelTrainer
from utils.losses import NonZeroEllipticBBoxLoss

if __name__ == '__main__':
    train_ds, test_ds = get_ds_pipeline()
    defect_cls = DefectLocalizeModel(backbone=tf.keras.applications.efficientnet.EfficientNetB0)
    trainer = ModelTrainer(defect_cls.model, epoch=10)
    trainer.set_optimizer(tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9, nesterov=True))
    trainer.set_loss(NonZeroEllipticBBoxLoss())
    trainer.set_metrics(tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.BinaryAccuracy())
    trainer.training_loop(train_ds, test_ds)
