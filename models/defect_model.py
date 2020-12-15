import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Model


class DefectLocalizeModel:
    def __init__(self, backbone: tf.keras.Model, num_output=9, input_shape=(224, 224, 3), weights='imagenet'):
        self.weights = weights
        self.input_shape = input_shape
        self.num_output = num_output
        self.backbone = backbone(input_shape=self.input_shape, include_top=False, weights=self.weights)
        self.backbone.trainable = False
        self._model = self.get_model()

    def get_model(self):
        x_1 = GlobalAvgPool2D()(self.backbone.output)
        x = BatchNormalization()(x_1)
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x_2 = ReLU()(x)
        x = Dropout(0.2)(x_2)
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x_3 = ReLU()(x)

        x = Dropout(0.2)(x_3)
        x_out_1 = Dense(1, activation='sigmoid', name='is_def')(x)

        x = Dropout(0.2)(x_3)
        x_out_2 = Dense(3, activation='softmax', name='cls')(x)

        x = Dropout(0.2)(x_2)
        x_out_3 = Dense(5, activation='sigmoid', name='bbox')(x)

        return Model(inputs=self.backbone.input, outputs=[x_out_1, x_out_3, x_out_2])

    @property
    def model(self):
        return self._model


if __name__ == '__main__':
    defect_cls = DefectLocalizeModel(backbone=tf.keras.applications.efficientnet.EfficientNetB0)
    defect_cls.model.summary()
