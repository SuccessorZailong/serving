import numpy as np
import tensorflow as tf

print(tf.__version__)


class SSQModel(tf.Module):
    def __init__(self):
        """do nothing"""
        super().__init__(name="ssq")

        self.r1_max = 17
        print("just test")

    def gen_red(self, r1):
        """gen_red on history distribution"""
        r1_max = self.r1_max
        r1 = tf.squeeze(r1)
        r1_max = tf.squeeze(r1_max)

        r1 = tf.cond(
            pred=r1 < r1_max,
            true_fn=lambda: r1,
            false_fn=lambda: tf.random.uniform(shape=[], minval=0, maxval=17, dtype=tf.int32)
        )

        # Use tf.random.uniform for generating random integers within specified ranges
        r2 = tf.random.uniform(shape=[], minval=r1 + 1, maxval=24, dtype=tf.int32)
        r3 = tf.random.uniform(shape=[], minval=r2 + 1, maxval=29, dtype=tf.int32)
        r4 = tf.random.uniform(shape=[], minval=r3 + 1, maxval=31, dtype=tf.int32)
        r5 = tf.random.uniform(shape=[], minval=r4 + 1, maxval=32, dtype=tf.int32)
        r6 = tf.random.uniform(shape=[], minval=r5 + 1, maxval=33, dtype=tf.int32)

        # Stack the results into a single tensor and add 1 to match the original function's behavior
        return tf.stack([r1, r2, r3, r4, r5, r6]) + 1

    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.int32)])
    def predict(self, x):
        """predict"""
        return self.gen_red(x)

    def get_serving_signatures(self):
        return {
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: self.predict,
        }


if __name__ == "__main__":
    ssq = SSQModel()
    version = "1"
    result = ssq.predict(7)
    print(result.numpy())

    tf.saved_model.save(ssq, f"./saved_model_{ssq.name}/{version}",
                        signatures=ssq.get_serving_signatures())
