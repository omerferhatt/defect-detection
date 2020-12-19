import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Concatenate, Add
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Model


class DefectLocalizeModel:
    def __init__(self, backbone: tf.keras.Model, input_shape=(224, 224, 3), weights='imagenet'):
        # Attributes
        self.weights = weights
        self.input_shape = input_shape
        # Get backbone
        self.backbone = backbone(input_shape=self.input_shape, include_top=False, weights=self.weights)
        # Freeze all layers of backbone
        self.backbone.trainable = False
        self._model = self.get_model()

    def get_model(self):
        # Backbone GAP output
        x_1 = GlobalAvgPool2D()(self.backbone.output)
        # FCN block - 1
        x = BatchNormalization()(self.backbone.output)
        x = Conv2D(64, (7, 7))(x)
        x = BatchNormalization()(x)
        y_1 = ReLU()(x)

        # FCN block for bbox center point regression
        x = Conv2D(32, (1, 1))(y_1)
        x = BatchNormalization()(x)
        x_center = ReLU()(x)
        x_center_to_is_def = Flatten()(x_center)
        x = Conv2D(2, (1, 1), activation='sigmoid')(x_center)
        x_out_4 = Flatten(name='bbox_center')(x)

        # FCN block for for bbox parametric regression
        x = Conv2D(32, (1, 1))(y_1)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x_param_to_is_def = Flatten()(x)
        # Information from center point concatenated
        x = Concatenate()([x, x_center])
        x = BatchNormalization()(x)
        x = Conv2D(16, (1, 1))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(3, (1, 1), activation='sigmoid')(x)
        x_out_3 = Flatten(name='bbox_param')(x)

        # Image classification node
        x = Dropout(0.5)(x_1)
        x_out_2 = Dense(3, activation='softmax', name='cls')(x)

        # Is defected classification node
        x_drop = Dropout(0.5)(x_1)
        x_add = Add()([x_center_to_is_def, x_param_to_is_def])
        x = Concatenate()([x_drop, x_out_2, x_add])
        x_out_1 = Dense(1, activation='sigmoid', name='is_def')(x)

        return Model(inputs=self.backbone.input, outputs=[x_out_1, x_out_2, x_out_3, x_out_4], name='defect_model')

    @property
    def model(self):
        return self._model


if __name__ == '__main__':
    defect_cls = DefectLocalizeModel(backbone=tf.keras.applications.efficientnet.EfficientNetB0)
    defect_cls.model.summary()