import tensorflow as tf


class NonZeroMSELoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        """
        Tries to eliminate effect of non defected image bbox values.
        Empty images have (0, 0, 0) parametric values. Mean squared error calculated
        between semi-major, semi-minor and rotation angle
        """
        # Finding zero summed row indices
        non_zero_ind = tf.squeeze(tf.where(tf.not_equal(tf.reduce_sum(y_true, 1), 0)))
        # Trimming indices
        y_true = tf.gather(y_true, non_zero_ind)
        y_pred = tf.gather(y_pred, non_zero_ind)
        mse_loss = tf.keras.losses.mse(y_true, y_pred)
        return mse_loss


class NonZeroL2Loss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        """
        Tries to eliminate effect of non defected image bbox values.
        Empty images have (0, 0) center values. L2 loss calculated between x-center and
        y-center.
        """
        # Finding zero summed row indices
        non_zero_ind = tf.squeeze(tf.where(tf.not_equal(tf.reduce_sum(y_true, 1), 0)))
        # Trimming indices
        y_true = tf.gather(y_true, non_zero_ind)
        y_pred = tf.gather(y_pred, non_zero_ind)
        return tf.reduce_mean(tf.square(y_true - y_pred))
