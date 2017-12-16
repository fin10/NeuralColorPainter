import os

import imageio
import tensorflow as tf

from paths import Paths

tf.logging.set_verbosity(tf.logging.INFO)


class ColorPaintingModel:

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
            num_epochs=epoch
        )

        reader = tf.WholeFileReader()
        _, image_file = reader.read(queue)
        image = tf.image.decode_jpeg(image_file, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)

        target = tf.image.decode_jpeg(image_file, channels=3)
        target = tf.image.convert_image_dtype(target, tf.float32)

        features = tf.stack([image])
        target = tf.stack([target])

        return features, target

    def __model_fn(self, features, labels, mode, params):
        learning_rate = params['learning_rate']

        inputs = features

        conv = tf.layers.conv2d(
            inputs=inputs,
            filters=3,
            kernel_size=[32, 32],
            padding='same'
        )

        predictions = conv

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
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=loss,
                global_step=tf.train.get_global_step(),
            )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=tf.image.convert_image_dtype(predictions, tf.uint8),
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
            input_fn=lambda: self.__input_fn(epoch=1),
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
