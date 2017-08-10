import tensorflow as tf, numpy as np, keras as K, sys
from IPython import embed
from scipy import misc
import misc_utils as MU
import action_enums as aenum

class AbstractModel(object):
    def __init__(self, modelfile, meanfile):
        MU.keras_model_serialization_bug_fix()
        self.model = K.models.load_model(modelfile) # this var stores the Keras Model
        self.mean = np.load(meanfile)

    def _preprocess_one(self, cur_frame): # do the same proprecessing as the one when the model was trained
        raise NotImplementedError("Override this method in derived classes.")

    def predict_one(self, cur_frame):
        raise NotImplementedError("Override this method in derived classes.")

    def _preprocess_one_default(self, img_np): 
        img_np = np.dot(img_np, [0.299, 0.587, 0.114]) # convert to grey scale
        img_np = misc.imresize(img_np, [84, 84], interp='bilinear')
        img_np = np.expand_dims(img_np, axis=2)
        img_np = img_np.astype(np.float32) / 255.0
        img_np -= self.mean
        return img_np

class BaselineModel(AbstractModel):
  def __init__(self, modelfile, meanfile):    
    super(BaselineModel, self).__init__(modelfile, meanfile)
  
  def _preprocess_one(self, img_np):
    return self._preprocess_one_default(img_np)

  def predict_one(self, img_np):
    img_np = self._preprocess_one(img_np)
    img_np = np.expand_dims(img_np, axis=0)
    logits = self.model.predict(img_np)[0] # returns a size-2 list where element [0] are the logits
    assert len(logits.shape)==2 and (logits.dtype==np.float32 or logits.dtype==np.float64), "simple sanity check: pred should be a vector containing probabilities of each action"
    return {"gaze": None, "raw_logits": logits}


class PastKFrameModel(AbstractModel):
  def __init__(self, modelfile, meanfile, K, stride, before):    
    super(PastKFrameModel, self).__init__(modelfile, meanfile)
    self.K, self.stride, self.before = int(K), int(stride), int(before)
    self.MAX_LEN = self.K * self.stride + self.before + 5 # + 5 does not mean anything, just store more memory for safety.
    self.frame_buffer = []
    self.DEFAULT_LOGITS_BEFORE_BUFFER_IS_FULL = ( # the model cannot work before K frames are seen, but needs to output something.
        np.array([0.0] * int(self.model.output[0].shape[1]))  # a hacky way to get number of classes of the model
        .reshape([1,-1]))
  
  def _preprocess_one(self, img_np):
    return self._preprocess_one_default(img_np)

  def predict_one(self, img_np):
    img_np = self._preprocess_one(img_np)
    img_np = np.expand_dims(img_np, axis=0)
    self.frame_buffer.append(img_np)

    if len(self.frame_buffer) <= self.MAX_LEN: # frame_buffer is not full yet, not ready to give a valid past K frame input to the network
        logits = self.DEFAULT_LOGITS_BEFORE_BUFFER_IS_FULL
    else:
        self.frame_buffer.pop(0)
        len_ = len(self.frame_buffer)-1
        pastKframes = self.frame_buffer[len_-self.before : len_-self.before-self.K*self.stride : -self.stride]
        img_np = np.concatenate(pastKframes,axis=-1)
        assert img_np.shape[-1] == self.K, "simple sanity check: the number of extracted past frames should be K"

        logits = self.model.predict(img_np)[0] # returns a size-2 list where element [0] are the logits
        assert len(logits.shape)==2 and (logits.dtype==np.float32 or logits.dtype==np.float64), "simple sanity check: pred should be a vector containing probabilities of each action"
    return {"gaze": None, "raw_logits": logits}


# TODO (Remark) To implement other models than the baseline model, those models might require different preprocessing of input.
# TODO So we need to create a different class for each different model.