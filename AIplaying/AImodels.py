import tensorflow as tf, numpy as np, keras as K, sys
from IPython import embed
from scipy import misc
import misc_utils as MU

class AbstractModel(object):
    def __init__(self, modelfile, meanfile):
        MU.keras_model_serialization_bug_fix()
        self.model = K.models.load_model(modelfile) # this var stores the Keras Model
        self.mean = np.load(meanfile)

    def _preprocess_one(self, cur_frame):
        raise NotImplementedError("Override this method in derived classes.")

    def predict_one(self, cur_frame):
        raise NotImplementedError("Override this method in derived classes.")

class BaselineModel(AbstractModel):
  def __init__(self, modelfile, meanfile):    
    super(BaselineModel, self).__init__(modelfile, meanfile)
  
  def _preprocess_one(self, img_np): # do the same proprecessing as the one when the model was trained
    img_np = np.dot(img_np, [0.299, 0.587, 0.114]) # convert to grey scale
    img_np = misc.imresize(img_np, [84, 84], interp='bilinear')
    img_np = np.expand_dims(img_np, axis=2)
    img_np = img_np.astype(np.float32) / 255.0
    img_np -= self.mean
    return img_np

  def predict_one(self, img_np):
    img_np = self._preprocess_one(img_np)
    img_np = np.expand_dims(img_np, axis=0)
    logits = self.model.predict(img_np)[0] # returns a size-2 list where element [0] are the logits
    assert len(logits.shape)==2 and (logits.dtype==np.float32 or logits.dtype==np.float64), "simple sanity check: pred should be a vector containing probabilitiesi of each action"
    action = np.argmax(logits[0,:])
    return {"action": action, "gaze": None, "raw_logits": logits}



# TODO (Remark) To implement other models than the baseline model, those models might require different preprocessing of input.
# TODO So we need to create a different class for each different model.