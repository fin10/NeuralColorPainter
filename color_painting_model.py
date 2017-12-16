import os

import imageio
import tensorflow as tf

from paths import Paths

tf.logging.set_verbosity(tf.logging.INFO)


class ColorPaintingModel:
    WIDTH = 200
    HEIGHT = 200

    def __init__(self):
        pass

    def __create_estimator(self):
        return tf.estimator.Estimator(
            model_fn=self.__model_fn,
            model_dir=Paths.MODEL,
            config=tf.estimator.RunConfig(
                save_summary_steps=10,
                save_checkpoints_steps=10,
            ),
            params={
                'learning_rate': 0.001
            }
        )

    def __input_fn(self, epoch: int = 1):
        queue = tf.train.string_input_producer(
            [os.path.join(Paths.IMAGES, 'image1.jpg')],
            num_epochs=epoch if epoch > 0 else None
        )

        reader = tf.WholeFileReader()
        _, image_file = reader.read(queue)
        image = tf.image.decode_jpeg(image_file, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)

        target = tf.image.decode_jpeg(image_file, channels=3)
        target = tf.image.convert_image_dtype(target, tf.float32)
        target = tf.image.rgb_to_hsv(target)

        images = tf.stack([image])
        targets = tf.stack([target])

        features = tf.image.resize_image_with_crop_or_pad(images, self.WIDTH, self.HEIGHT)
        targets = tf.image.resize_image_with_crop_or_pad(targets, self.WIDTH, self.HEIGHT)

        return features, targets

    def __model_fn(self, features, labels, mode, params):
        learning_rate = params['learning_rate']

        inputs = features

        def conv2d(inputs, filters, kernel_size, strides):
            return tf.layers.conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )

        outputs = conv2d(inputs, 64, [3, 3], [2, 2])
        outputs = conv2d(outputs, 128, [3, 3], [1, 1])

        outputs = conv2d(outputs, 128, [3, 3], [2, 2])
        outputs = conv2d(outputs, 256, [3, 3], [1, 1])

        outputs = conv2d(outputs, 256, [3, 3], [2, 2])
        outputs = conv2d(outputs, 512, [3, 3], [1, 1])

        outputs = conv2d(outputs, 512, [3, 3], [1, 1])
        outputs = conv2d(outputs, 256, [3, 3], [1, 1])
        outputs = conv2d(outputs, 128, [3, 3], [1, 1])

        outputs = tf.reshape(outputs, [-1, int(self.WIDTH / 4) * int(self.HEIGHT / 4) * 32])
        outputs = tf.reshape(outputs, [-1, int(self.WIDTH / 2), int(self.HEIGHT / 2), 8])
        outputs = conv2d(outputs, 64, [3, 3], [1, 1])

        outputs = tf.reshape(outputs, [-1, int(self.WIDTH / 2) * int(self.HEIGHT / 2) * 64])
        outputs = tf.reshape(outputs, [-1, self.WIDTH, self.HEIGHT, 16])
        outputs = conv2d(outputs, 32, [3, 3], [1, 1])
        outputs = conv2d(outputs, 2, [3, 3], [1, 1])

        predictions = tf.concat([outputs, inputs], 3)

        loss = None
        if mode != tf.estimator.ModeKeys.PREDICT:
            loss = tf.losses.mean_squared_error(
                labels=labels,
                predictions=predictions
            )

        eval_metric_ops = None
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {
                'rmse': tf.metrics.root_mean_squared_error(labels, predictions),
            }

        train_op = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            learning_rate = tf.train.exponential_decay(
                learning_rate=learning_rate,
                global_step=tf.train.get_global_step(),
                decay_steps=100,
                decay_rate=0.96
            )

            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=loss,
                global_step=tf.train.get_global_step(),
            )

        predictions = tf.image.hsv_to_rgb(predictions)
        predictions = tf.image.convert_image_dtype(predictions, tf.uint8)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops
        )

    def train(self):
        class ValidationHook(tf.train.SessionRunHook):

            def __init__(self, estimator: tf.estimator.Estimator, input_fn):
                self.__every_n_steps = 100
                self.__estimator = estimator
                self.__input_fn = input_fn

            def before_run(self, run_context):
                graph = run_context.session.graph
                return tf.train.SessionRunArgs(tf.train.get_global_step(graph))

            def after_run(self, run_context, run_values):
                if run_values.results % self.__every_n_steps == 0:
                    result = self.__estimator.evaluate(
                        input_fn=lambda: self.__input_fn(),
                    )
                    print('#%d %s' % (run_values.results, result))

        estimator = self.__create_estimator()
        estimator.train(
            input_fn=lambda: self.__input_fn(epoch=-1),
            hooks=[ValidationHook(estimator, self.__input_fn)]
        )

    def color(self):
        estimator = self.__create_estimator()
        result = list(estimator.predict(
            input_fn=lambda: self.__input_fn(),
        ))[0]

        if not os.path.exists(Paths.OUTPUT):
            os.mkdir(Paths.OUTPUT)

        with open(os.path.join(Paths.OUTPUT, './output.jpg'), mode='w') as fp:
            imageio.imwrite(fp, result, format='jpg')
