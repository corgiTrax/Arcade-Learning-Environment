import tensorflow as tf, numpy as np, keras as K
import shutil, os, time, re
from IPython import embed
import ipdb

def acc(y_true, y_pred):
  return tf.reduce_mean(
    tf.cast(tf.nn.in_top_k(
      targets=tf.squeeze(tf.cast(y_true,tf.int32)), 
      predictions=y_pred,k=1),tf.float32))

def save_GPU_mem_keras():
    # don't let tf eat all the memory on eldar-11
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.backend.set_session(sess)


class ExprCreaterAndResumer:
    def __init__(self, rootdir):
        if not os.path.exists(rootdir):
            os.makedirs(rootdir)
        expr_dirs = os.listdir(rootdir)
        indices = [int(re.match("(\d+)_", x).group(1)) for x in expr_dirs]
        highest_indices = max(indices) if len(indices)>0 else -1

        self.dir_lasttime = rootdir + '/' + expr_dirs[highest_indices] if highest_indices != -1 else None
        # dir name is like "5_Mar-09-12-27-59"
        self.dir = rootdir + '/' +  str(highest_indices+1) + \
            '_' + time.strftime("%b-%d-%H-%M-%S")
        os.mkdir(self.dir)
        self.logfile = open(self.dir +"/log.txt", 'a')

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

    def save_weight_and_training_config_state(self, model):
        model.save(self.dir + '/model.hdf5')

    def printdebug(self, str):
        print('  ----   DEBUG: '+str)
        self.logfile.write('  ----   DEBUG: '+str+'\n')
        self.logfile.flush()
 
def serialize_model_keras_bug_fix(obj_to_serialize):
    from keras.utils.generic_utils import get_custom_objects
    get_custom_objects().update({obj_to_serialize.__name__: obj_to_serialize})