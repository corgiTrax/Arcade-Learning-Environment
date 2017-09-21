import tensorflow as tf, numpy as np, keras as K, cv2, sys
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
        img_np = np.expand_dims(img_np, axis=0)
        return img_np

class BaselineModel(AbstractModel):
  def __init__(self, modelfile, meanfile):    
    super(BaselineModel, self).__init__(modelfile, meanfile)
  
  def _preprocess_one(self, img_np):
    return self._preprocess_one_default(img_np)

  def predict_one(self, img_np):
    img_np = self._preprocess_one(img_np)
    logits = self.model.predict(img_np)[0] # returns a size-2 list where element [0] are the logits
    assert len(logits.shape)==2 and (logits.dtype==np.float32 or logits.dtype==np.float64), "simple sanity check: pred should be a vector containing probabilities of each action"
    return {"gaze": None, "raw_logits": logits}


class PastKFrameModel(AbstractModel):
  def __init__(self, modelfile, meanfile, k, stride, before):
    super(PastKFrameModel, self).__init__(modelfile, meanfile)
    self.k, self.stride, self.before = int(k), int(stride), int(before)
    self.MAX_LEN = self.k * self.stride + self.before + 5 # + 5 does not mean anything, just store more memory for safety.
    self.frame_buffer = []
    self.DEFAULT_LOGITS_BEFORE_BUFFER_IS_FULL = ( # the model cannot work before K frames are seen, but needs to output something.
        np.array([0.0] * int(self.model.output[0].shape[1]))  # a hacky way to get number of classes of the model
        .reshape([1,-1]))

  def extract_and_pop_buffer(self, buf):
    buf.pop(0)
    len_ = len(buf)-1
    pastKframes = buf[len_-self.before : len_-self.before-self.k*self.stride : -self.stride]
    img_np = np.concatenate(pastKframes,axis=-1)
    assert img_np.shape[-1] == self.k, "simple sanity check: the number of extracted past frames should be K. shape of img_np: %s" % str(img_np.shape)
    return img_np
  
  def _preprocess_one(self, img_np):
    return self._preprocess_one_default(img_np)

  def predict_one(self, img_np):
    img_np = self._preprocess_one(img_np)
    self.frame_buffer.append(img_np)

    if len(self.frame_buffer) <= self.MAX_LEN: # frame_buffer is not full yet, not ready to give a valid past K frame input to the network
        logits = self.DEFAULT_LOGITS_BEFORE_BUFFER_IS_FULL
    else:
        img_np = self.extract_and_pop_buffer(self.frame_buffer)
        logits = self.model.predict(img_np)[0] # returns a size-2 list where element [0] are the logits
        assert len(logits.shape)==2 and (logits.dtype==np.float32 or logits.dtype==np.float64), "simple sanity check: pred should be a vector containing probabilities of each action"
    return {"gaze": None, "raw_logits": logits}

# This model first predicts gaze using past K frames and then multiply the gaze heap map with the current frame
class PastKFrameGaze_and_CurrentFrameAction(PastKFrameModel): 
  def __init__(self, modelfile, meanfile, k, stride, before, gaze_pred_modelfile):
    super(PastKFrameGaze_and_CurrentFrameAction, self).__init__(modelfile, meanfile, k, stride, before)
    self.gaze_pred_model = K.models.load_model(gaze_pred_modelfile)
  
  def _preprocess_one(self, img_np):
    return self._preprocess_one_default(img_np)

  def predict_one(self, img_np):
    img_np = self._preprocess_one(img_np)
    self.frame_buffer.append(img_np)

    if len(self.frame_buffer) <= self.MAX_LEN: # frame_buffer is not full yet, not ready to give a valid past K frame input to the network
        logits = self.DEFAULT_LOGITS_BEFORE_BUFFER_IS_FULL
        GHmap = None
    else:
        img_np = self.extract_and_pop_buffer(self.frame_buffer)
        GHmap = self.gaze_pred_model.predict(img_np)
        logits = self.model.predict([img_np[...,-2:-1], GHmap])[0] # returns a size-2 list where element [0] are the logits
        assert len(logits.shape)==2 and (logits.dtype==np.float32 or logits.dtype==np.float64), "simple sanity check: pred should be a vector containing probabilities of each action"
    return {"gaze": GHmap, "raw_logits": logits}


# This model first predicts gaze using past K frames and then multiply the gaze heap map with the current frame
class PastKFrameOpticalFlowGaze_and_CurrentFrameAction(PastKFrameModel):  
  def __init__(self, modelfile, meanfile, k, stride, before,
                     gaze_pred_modelfile, optical_flow_meanfile):
    super(PastKFrameOpticalFlowGaze_and_CurrentFrameAction, self).__init__(modelfile, meanfile, k, stride, before)
    self.gaze_pred_model = K.models.load_model(gaze_pred_modelfile)
    self.optical_flow_mean = np.load(optical_flow_meanfile)
    self.opticalf_buffer = []
    
    from collections import deque
    self.length2_buffer_for_computing_optical_flow = deque(maxlen=2)

    self.predicted_count = 0 # stored how many times predict_one () is called
  
  def _preprocess_one(self, img_np):
    return self._preprocess_one_default(img_np)

  def predict_one(self, img_np):
    self.length2_buffer_for_computing_optical_flow.append(np.dot(img_np, [0.299, 0.587, 0.114]))
    img_np = self._preprocess_one(img_np)
    self.frame_buffer.append(img_np)

    # Compute optical flow and append it to buffer
    if len(self.frame_buffer) > 1:
        prev = self.length2_buffer_for_computing_optical_flow[0]
        cur = self.length2_buffer_for_computing_optical_flow[-1]
        flow_ = cv2.calcOpticalFlowFarneback(prev, cur,
                    None, 0.5, 3, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN or
                    cv2.OPTFLOW_USE_INITIAL_FLOW)
        # compute magnitude
        fx, fy = flow_[:,:,0], flow_[:,:,1]
        flow = np.sqrt(fx*fx+fy*fy)
        flow = cv2.normalize(flow, None, 0, 1, cv2.NORM_MINMAX)
        flow = cv2.resize(flow, (84, 84))
        flow -= self.optical_flow_mean
        self.opticalf_buffer.append(flow[np.newaxis, ..., np.newaxis])
    else:
        self.opticalf_buffer.append(None) # Appending None is needed to align with self.frame_buffer

    if len(self.frame_buffer) <= self.MAX_LEN: # frame_buffer is not full yet, not ready to give a valid past K frame input to the network
        logits = self.DEFAULT_LOGITS_BEFORE_BUFFER_IS_FULL
        GHmap = None
    else:
        img_np = self.extract_and_pop_buffer(self.frame_buffer)
        flow_np = self.extract_and_pop_buffer(self.opticalf_buffer)
        GHmap = self.gaze_pred_model.predict([img_np, flow_np])
        logits = self.model.predict([img_np[...,-2:-1], GHmap])[0] # returns a size-2 list where element [0] are the logits
        assert len(logits.shape)==2 and (logits.dtype==np.float32 or logits.dtype==np.float64), "simple sanity check: pred should be a vector containing probabilities of each action"

    self.predicted_count += 1
    return {"gaze": GHmap, "raw_logits": logits}


# TODO (Remark) To implement other models than the baseline model, those models might require different preprocessing of input.
# TODO So we need to create a different class for each different model.
