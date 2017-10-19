import tensorflow as tf, numpy as np, keras as K
import shutil, os, time, re
from IPython import embed


# Example usage : color("WARN:', 'red') returns a red string 'WARN:' which you can print onto terminal
def color(str_, color):
    return getattr(Colors,color.upper())+str(str_)+Colors.RESET
class Colors:
    RED   = "\033[1;31m"
    BLUE  = "\033[1;34m"
    CYAN  = "\033[1;36m"
    GREEN = "\033[0;32m"
    RESET = "\033[0;0m"
    BOLD    = "\033[;1m"
    REVERSE = "\033[;7m"

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

        # dir name is like "5_Mar-09-12-27-59" or "5_<postfix>"
        self.dir = rootdir + '/' +  '%02d' % (expr_num[highest_idx]+1 if highest_idx != -1 else 0) + \
            '_' + (postfix if postfix else time.strftime("%b-%d-%H-%M-%S") )
        os.makedirs(self.dir)
        self.logfile = open(self.dir +"/log.txt", 'a', 0) # no buffer
        self.redirect_output_to_logfile_as_well()

    def load_weight_and_training_config_and_state(self, model_file_path):
        """
            Call keras.models.load_model(fname) to load the arch, weight, 
            training states and config (loss, optimizer) of the model.
            Note that model.load_weights() and keras.models.load_model() are different.
            model.load_weights() just loads weight, and is not used here.
        """
        return K.models.load_model(model_file_path)

    def dump_src_code_and_model_def(self, fname, kerasmodel):
        fname = os.path.abspath(fname) # if already absolute path, it does nothing
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
        print 'redirect_output_to_logfile_if_not_on() is deprecated. Please delete the line that calls it.'
        print 'This func still exists because old code might use it.'

    def redirect_output_to_logfile_as_well(self):
        class Logger(object): 
            def __init__(self, logfile):
                self.stdout = sys.stdout
                self.logfile = logfile
            def write(self, message):
                self.stdout.write(message)
                self.logfile.write(message)
            def flush(self):
                #this flush method is needed for python 3 compatibility.
                #this handles the flush command by doing nothing.
                #you might want to specify some extra behavior here.
                pass
        sys.stdout = Logger(self.logfile)
        sys.stderr = sys.stdout
        # Now you can use: `print "Hello"`, which will write "Hello" to both stdout and logfile

    def printdebug(self, str):
        print('  ----   DEBUG: '+str)

class PrintLrCallback(K.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        print ("lr: %f" % K.backend.get_value(self.model.optimizer.lr))

