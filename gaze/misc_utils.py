import tensorflow as tf, numpy as np, keras as K
import shutil, os, time, re
from IPython import embed
import ipdb

def save_GPU_mem_keras():
    # don't let tf eat all the memory on eldar-11
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.backend.set_session(sess)

class ExprCreaterAndResumer:
    def __init__(self, rootdir, postfix=None):
        if not os.path.exists(rootdir):
            os.makedirs(rootdir)
        expr_dirs = os.listdir(rootdir)
        re_matches = [re.match("(\d+)_", x) for x in expr_dirs]
        expr_num = [int(x.group(1)) for x in re_matches if x is not None]
        highest_idx = np.argmax(expr_num) if len(expr_num)>0 else -1

        self.dir_lasttime = "%s/%s" % (rootdir, expr_dirs[highest_idx]) if highest_idx != -1 else None
        # dir name is like "5_Mar-09-12-27-59"
        self.dir = rootdir + '/' +  str(expr_num[highest_idx]+1 if highest_idx != -1 else 0) + \
            '_' + (postfix if postfix else time.strftime("%b-%d-%H-%M-%S") )
        os.mkdir(self.dir)
        self.logfile = open(self.dir +"/log.txt", 'a', 0) # no buffer

    def load_weight_and_training_config_and_state(self):
        """
            Call keras.models.load_model(fname) to load the arch, weight, 
            training states and config (loss, optimizer) of the model.
            Note that model.load_weights() and keras.models.load_model() are different.
            model.load_weights() just loads weight, and is not used here.
        """
        if self.dir_lasttime is None: raise ValueError("Directory which stores the model is not found.")
        fname = self.dir_lasttime + '/model.hdf5'
        return K.models.load_model(fname)

    def dump_src_code_and_model_def(self, fname, kerasmodel):
        shutil.copyfile(fname, self.dir + '/' + os.path.basename(fname))
        with open(self.dir + '/model.yaml', 'w') as f:
            f.write(kerasmodel.to_yaml())
        # copy all py files 
        snapshot_dir = self.dir + '/all_py_files_snapshot'
        os.makedirs(snapshot_dir)
        py_files = [os.path.dirname(fname)+'/'+x for x in os.listdir(os.path.dirname(fname)) if x.endswith('.py')]
        for py in py_files:
            shutil.copyfile(py, snapshot_dir + '/' + os.path.basename(py))

    def save_weight_and_training_config_state(self, model):
        model.save(self.dir + '/model.hdf5')

    def redirect_output_to_logfile_if_not_on(self, hostname):
        import socket, sys
        if socket.gethostname() != hostname:
            sys.stdout, sys.stderr = self.logfile, self.logfile

    def printdebug(self, str):
        print('  ----   DEBUG: '+str)
        self.logfile.write('  ----   DEBUG: '+str+'\n')
        self.logfile.flush()
 
def keras_model_serialization_bug_fix(): # stupid keras
    from keras.utils.generic_utils import get_custom_objects
    f=lambda obj_to_serialize: \
        get_custom_objects().update({obj_to_serialize.__name__: obj_to_serialize})
    f(loss_func)
    f(acc_)

def loss_func(target, pred): 
    return K.backend.sparse_categorical_crossentropy(output=pred, target=target, from_logits=True)

def acc_(y_true, y_pred): # don't rename it to acc or accuracy (otherwise stupid keras will replace this func with its own accuracy function when serializing )
  return tf.reduce_mean(
    tf.cast(tf.nn.in_top_k(
      targets=tf.squeeze(tf.cast(y_true,tf.int32)), 
      predictions=y_pred,k=1),tf.float32))

def my_softmax(x):
    """Softmax activation function. Normalize the whole metrics.
    # Arguments
        x : Tensor.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
   
    return K.activations.softmax(x, axis=[1,2,3])

def my_kld(y_true, y_pred):
    """
    Correct keras bug. Compute the KL-divergence between two metrics.
    """
    y_true = K.backend.clip(y_true, K.backend.epsilon(), 1)
    y_pred = K.backend.clip(y_pred, K.backend.epsilon(), 1)
    return K.backend.sum(y_true * K.backend.log(y_true / y_pred), axis = [1,2,3])

def NSS(y_true, y_pred):
    """
    This function is to calculate the NSS score of the predict saliency map.

    Input: y_true: ground truth saliency map
           y_pred: predicted saliency map

    Output: NSS score. float num
    """
    
    stddev = tf.contrib.keras.backend.std(y_pred, axis = [1,2,3])
    stddev = tf.expand_dims(stddev, 1)
    stddev = tf.expand_dims(stddev, 2)
    stddev = tf.expand_dims(stddev, 3)
    mean = tf.reduce_mean(y_pred, axis = [1,2,3], keep_dims=True)
    sal = (y_pred - mean) / stddev
    score = tf.multiply(y_true, sal)
    score = tf.contrib.keras.backend.sum(score, axis = [1,2,3])
    
    return score

class PrintLrCallback(K.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        print ("lr: %f" % K.backend.get_value(self.model.optimizer.lr))
