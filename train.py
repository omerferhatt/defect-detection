import time

import tensorflow as tf


class ModelTrainer:
    def __init__(self, model: tf.keras.Model, epoch: int):
        self.model = model
        self.epoch = epoch

        self._optimizer = None
        self._loss = None
        self._train_metrics = None
        self._val_metrics = None

    # @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            is_defect_y = y[0]
            bbox_y = y[1]
            cls_y = y[2]
            non_zero_ind = tf.squeeze(tf.where(tf.equal(tf.reduce_sum(is_defect_y, 1), 1)))
            bbox_y = tf.gather(bbox_y, non_zero_ind)
            logits[1] = tf.gather(logits[1], non_zero_ind)
            loss_is_defect = tf.reduce_mean(tf.keras.losses.binary_crossentropy(is_defect_y, logits[0]))
            loss_bbox = self._loss(bbox_y, logits[1])
            loss_cls = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(cls_y, logits[2]))
            loss_value = tf.reduce_mean([loss_is_defect, loss_bbox, loss_cls])

        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self._optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self._train_metrics[0].update_state(cls_y, logits[2])
        self._train_metrics[1].update_state(is_defect_y, logits[0])
        return loss_value, loss_bbox

    @tf.function
    def test_step(self, x, y):
        val_logits = self.model(x, training=False)
        is_defect_y = y[0]
        cls_y = y[2]
        self._val_metrics[0].update_state(cls_y, val_logits[2])
        self._val_metrics[1].update_state(is_defect_y, val_logits[0])

    def training_loop(self, train_ds, test_ds):
        for epoch in range(self.epoch):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, *y_batch_train) in enumerate(train_ds):
                loss_value, loss_bbox = self.train_step(x_batch_train, y_batch_train)

                # Log every 200 batches.
                if step % 50 == 0:
                    print(
                        "Training loss and bounding box loss(for one batch) at step %d: %.4f %.4f"
                        % (step, float(loss_value), float(loss_bbox))
                    )
                    print("Seen so far: %d samples" % ((step + 1) * 32))

            # Display metrics at the end of each epoch.
            train_cls_acc = self._train_metrics[0].result()
            train_is_defect_acc = self._train_metrics[1].result()

            print("Training defect acc over epoch: %.4f" % (float(train_is_defect_acc),))
            print("Training classification acc over epoch: %.4f" % (float(train_cls_acc),))

            # Reset training metrics at the end of each epoch
            self._train_metrics[0].reset_states()
            self._train_metrics[1].reset_states()

            # Run a validation loop at the end of each epoch.
            for x_batch_val, *y_batch_val in test_ds:
                self.test_step(x_batch_val, y_batch_val)

            val_cls_acc = self._val_metrics[0].result()
            val_is_defect_acc = self._val_metrics[1].result()
            self._val_metrics[0].reset_states()
            self._val_metrics[1].reset_states()
            print("Validation defect acc: %.4f" % (float(val_is_defect_acc),))
            print("Validation classification acc: %.4f" % (float(val_cls_acc),))
            print("Time taken: %.2fs" % (time.time() - start_time))

    def set_optimizer(self, opt):
        assert isinstance(opt, tf.keras.optimizers.Optimizer)
        self._optimizer = opt

    def set_loss(self, loss):
        assert isinstance(loss, tf.keras.losses.Loss)
        self._loss = loss

    def set_metrics(self, *args):
        for arg in args:
            assert isinstance(arg, tf.keras.metrics.Metric)
        self._train_metrics = args
        self._val_metrics = args
