import tensorflow as tf


class NonZeroEllipticBBoxLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        l2_loss_center = tf.sqrt(tf.reduce_sum(tf.square(y_true[:, -2:], y_pred[:, -2:])))
        mse_loss_semi_ax = tf.reduce_mean(tf.square(tf.subtract(y_true[:, :2], y_pred[:, :2])))
        mae_loss_rad_deg = tf.reduce_mean(tf.abs(tf.subtract(y_true[:, 2:3], y_pred[:, 2:3])))
        total_loss = l2_loss_center * 0.5 + mse_loss_semi_ax * 0.3 + mae_loss_rad_deg * 0.2
        return total_loss
