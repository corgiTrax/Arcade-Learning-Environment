import tensorflow as tf, numpy as np, keras as K, sys
import keras.layers as L
from keras.models import Model, Sequential # keras/engine/training.py
from IPython import embed
import input_utils, misc_utils as MU
import ipdb

MU.BMU.save_GPU_mem_keras()
MU.keras_model_serialization_bug_fix()

try:
    model=K.models.load_model('../AIplaying/baseline_actionModels/seaquest.hdf5')
except:
    model=K.models.load_model('../AIplaying/baseline_actionModels/ECCV/seaquest.hdf5')

print("The goal of this file is to show keras model can be trained using native tensorflow api.")
print("Model Archetecture:")
model.summary()

# Define a convenient function to print the norm of the kernel of the softmax layer
# model.layers[-4] is the last layer 
get_last_layer_kernel_norm=lambda: np.linalg.norm(model.layers[-4].get_weights()[0])

print('Layer norm prior to training:')
print(get_last_layer_kernel_norm())

print("Now we begin to train the model using native tensorflow api.")
model_out = model.outputs[0] # the first output of that keras model, 
                             # defined at ale/modeling/pyModel/main-base.py
target=tf.placeholder(dtype=tf.float32,shape=(None,18))
loss=tf.reduce_mean(tf.square(model_out-target)) # mean squared error
train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(loss)
for i in range(2):
    train_step.run(
        session=K.backend.get_session(), 
        feed_dict={
            model.input: np.random.randn(1,84,84,1), 
            target: np.zeros([1,18]), 
            K.backend.learning_phase(): 1
        })
    print(get_last_layer_kernel_norm())

print('You can see the norm is changing, indicating that the layer is being trained.')
