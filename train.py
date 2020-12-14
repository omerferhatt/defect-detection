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

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            is_defect_y = y[:, :1]
            bbox_y = y[:, 1:5]
            cls_y = y[:, 5:]
            loss_is_defect = tf.keras.losses.binary_crossentropy(is_defect_y, logits[:, :1], from_logits=True)
            loss_bbox = tf.keras.losses.mse(bbox_y, logits[:, 1:5])
            loss_cls = tf.keras.losses.categorical_crossentropy(cls_y, logits[:, 5:], from_logits=True)
            loss_value = tf.reduce_mean([loss_is_defect, loss_bbox, loss_cls])

        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self._optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self._train_metrics[0].update_state(cls_y, logits[:, 5:])
        self._train_metrics[1].update_state(is_defect_y, logits[:, :1])
        return loss_value, tf.reduce_mean(loss_bbox, axis=-1)

    @tf.function
    def test_step(self, x, y):
        val_logits = self.model(x, training=False)
        is_defect_y = y[:, :1]
        cls_y = y[:, 5:]
        self._val_metrics[0].update_state(cls_y, val_logits[:, 5:])
        self._val_metrics[1].update_state(is_defect_y, val_logits[:, :1])

    def training_loop(self, train_ds, test_ds):
        for epoch in range(self.epoch):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
                loss_value, loss_bbox = self.train_step(x_batch_train, y_batch_train)

                # Log every 200 batches.
                if step % 200 == 0:
                    print(
                        "Training loss and bounding box loss(for one batch) at step %d: %.4f %.4f"
                        % (step, float(loss_value), float(loss_bbox))
                    )
                    print("Seen so far: %d samples" % ((step + 1) * 64))

            # Display metrics at the end of each epoch.
            train_cls_acc = self._train_metrics[0].result()
            train_is_defect_acc = self._train_metrics[1].result()

            print("Training defect acc over epoch: %.4f" % (float(train_is_defect_acc),))
            print("Training classification acc over epoch: %.4f" % (float(train_cls_acc),))

            # Reset training metrics at the end of each epoch
            self._train_metrics[0].reset_states()
            self._train_metrics[1].reset_states()

            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in test_ds:
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
