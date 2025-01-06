import numpy as np
import tensorflow as tf

print(tf.__version__)


class SSQModel(tf.Module):
    def __init__(self):
        """do nothing"""
        super().__init__(name="ssq")

        self.r1_max = 17
        print("just test")

    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.int32)])
    def predict(self, x):
        """predict"""

        def gen_red(r1_max, r1):
            """gen_red on history distribution"""
            if r1 < r1_max:
                print("r1 is ok")
            else:
                r1 = tf.constant(np.random.randint(0, 17))
            r2 = np.random.randint(r1 + 1, 24)
            r3 = np.random.randint(r2 + 1, 29)
            r4 = np.random.randint(r3 + 1, 31)
            r5 = np.random.randint(r4 + 1, 32)
            r6 = np.random.randint(r5 + 1, 33)
            return np.array([r1, r2, r3, r4, r5, r6]) + 1

        return tf.py_function(gen_red, [self.r1_max, x], tf.int32)

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
