import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L1L2


def get_model(input_shape=(224, 224, 3)):
    mobilenetv2 = EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')
    for layer in mobilenetv2.layers[-30:]:
        layer.trainable = True
    for layer in mobilenetv2.layers[:-30]:
        layer.trainable = False

    avg_pool = layers.GlobalAvgPool2D()(mobilenetv2.output)
    # Class type classification output node
    class_type_batch_norm_1 = layers.BatchNormalization()(avg_pool)
    class_type_dense_1 = layers.Dense(256, activation='relu', kernel_regularizer=L1L2(l1=0.001, l2=0.001))(
        class_type_batch_norm_1)
    class_type_batch_norm_2 = layers.BatchNormalization()(class_type_dense_1)
    class_type_dense_out = layers.Dense(3, activation='softmax', name='class')(class_type_batch_norm_2)
    # Is defected binary classification output node
    is_defect_batch_norm_1 = layers.BatchNormalization()(avg_pool)
    is_defect_dense_1 = layers.Dense(256, activation='relu', kernel_regularizer=L1L2(l1=0.001, l2=0.001))(
        is_defect_batch_norm_1)
    is_defect_batch_norm_2 = layers.BatchNormalization()(is_defect_dense_1)
    is_defect_dense_out = layers.Dense(1, activation='sigmoid', name='defect')(is_defect_batch_norm_2)
    # Ellipse bounding box localization output node
    ellipse_batch_norm_1 = layers.BatchNormalization()(avg_pool)
    ellipse_bbox_dense_1 = layers.Dense(256, activation='elu', kernel_regularizer=L1L2(l1=0.001, l2=0.001))(
        ellipse_batch_norm_1)
    ellipse_batch_norm_2 = layers.BatchNormalization()(ellipse_bbox_dense_1)
    ellipse_bbox_dense_out = layers.Dense(5, name='bbox')(ellipse_batch_norm_2)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-2)

    model = Model(inputs=mobilenetv2.input, outputs=[class_type_dense_out, is_defect_dense_out, ellipse_bbox_dense_out])
    model.compile(
        optimizer=opt,
        loss={'class': 'categorical_crossentropy',
              'defect': 'binary_crossentropy',
              'bbox': 'mse'},
        metrics={'class': tf.keras.metrics.CategoricalAccuracy(name='acc'),
                 'defect': tf.keras.metrics.BinaryAccuracy(name='acc'),
                 'bbox': tf.keras.metrics.MeanSquaredError(name='mse')})

    return model


if __name__ == '__main__':
    model = get_model((224, 224, 3))
    model.summary()
