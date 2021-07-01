import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from df_bundle import *
import tensorflow as tf
import numpy as np
import sys

# Compute model output from bundle and the corresponding input
def output_from_bundle(ws,x):
   x_ext = np.expand_dims(np.concatenate([np.ones((len(x), 1)), x], axis=1), axis=2)
   v = np.sum(x_ext * ws, axis=1)

   return v

sys.stdout.write("Loading dataset...");sys.stdout.flush()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

x_train = np.reshape(x_train,(len(x_train),-1))
x_test = np.reshape(x_test,(len(x_test),-1))
sys.stdout.write("done\n");sys.stdout.flush()

sys.stdout.write("Creating a model...");sys.stdout.flush()
nsamples, ninputs = x_train.shape
noutputs = len(np.unique(y_train))

model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=128,activation='relu',input_shape=(ninputs,)),
                            tf.keras.layers.Dense(units=32,activation='relu'),
                            tf.keras.layers.Dense(units=noutputs,activation='softmax')])
model.compile(
   loss='sparse_categorical_crossentropy',
   optimizer='adam',
   metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=10, batch_size=128,shuffle=True, verbose=False)

# Strip softmax
K = model.layers[-1].output.shape[1]
v = model.output.op.inputs[0]
model_without_softmax = tf.keras.models.Model(model.input, v)
sys.stdout.write("done\n");sys.stdout.flush()

N = 2000 #Number of points to sample from the training set
         #for the computation of conceptual capacity.
selected_indexes = np.random.permutation(len(x_test))[:N]
x_test = x_test[selected_indexes]

sys.stdout.write("Computing the df bundle for %d points..." % len(x_test));sys.stdout.flush()
ws = df_bundle(model,x_test)
sys.stdout.write("done\n");sys.stdout.flush()

v_predict_from_model = model_without_softmax.predict(x_test)
v_predict_from_ws = output_from_bundle(ws,x_test)

mean_error = np.sqrt(np.mean(np.square(v_predict_from_model-v_predict_from_ws)))

print("Mean difference between model predictions and bundle-based output = %.3f" % mean_error)

sys.stdout.write("Creating similarity matrix with tensordot...");sys.stdout.flush()
S_with_tensordot = df_similarity(x_test,ws,tensordot_datasize_limit=N+1)
sys.stdout.write("done\nCreating similarity matrix without tensordot...");sys.stdout.flush()
S_without_tensordot = df_similarity(x_test,ws,tensordot_datasize_limit=N-1)
sys.stdout.write("done\n");sys.stdout.flush()

mean_error = np.sqrt(np.mean(np.square(S_with_tensordot-S_without_tensordot)))

print("Mean difference between similarity with and without tensordot  = %.3f" % mean_error)

H_with_tensordot = df_von_Neumann_entropy(S_with_tensordot)
H_without_tensordot = df_von_Neumann_entropy(S_without_tensordot)

print("Hdf with    tensordot  = %.3f" % H_with_tensordot)
print("Hdf without tensordot  = %.3f" % H_without_tensordot)

