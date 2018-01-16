from tensorflow.python.training import session_run_hook
import tensorflow as tf

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        tf.summary.histogram('histogram', var)

class TensorboardViewHook(session_run_hook.SessionRunHook):
    """Adds new variables to the session to monitor in TensorBoard
    """

    def __init__(self):
        """Initializes a `NanTensorHook`.
        """
        print("\n\nInitializing TensorboardViewHook\n\n")

    def after_create_session(self, session, coord):
        print("\n\nAfter creating the session\n\n")


