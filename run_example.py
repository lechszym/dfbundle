from df_bundle import *
import tensorflow as tf
import numpy as np
from skimage.transform import resize
import sys

# Get the dataset
dataset = 'mnist'
sys.stdout.write("Loading %s dataset..." % dataset);sys.stdout.flush()
if dataset=='mnist':
   (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
   x_train = np.expand_dims(x_train, axis=3)
   x_test = np.expand_dims(x_test, axis=3)
elif dataset == 'cifar10':
   (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Upsample images to 48x48 pixels...
sys.stdout.write("done\nUpsampling images to 48x48...");sys.stdout.flush()
x_train = np.stack([resize(im, (48,48)) for im in x_train],axis=0)
x_test = np.stack([resize(im, (48,48)) for im in x_test],axis=0)

x_train = (x_train*255).astype('uint8')
x_test *= (x_test*255).astype('uint8')
sys.stdout.write("done\n");sys.stdout.flush()


# Load the VGG16 architecture
_, height, width,channels = x_train.shape
noutputs = len(np.unique(y_train))

model = tf.keras.applications.VGG16(
            include_top=True,
            weights=None,
            input_shape=(height,width,channels),
            pooling=None,
            classes=noutputs,
            classifier_activation="softmax",
        )

# Train the model
optimiser = tf.keras.optimizers.Adam(learning_rate=1e-4)

model.compile(
   loss='sparse_categorical_crossentropy',
   optimizer=optimiser,
   metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=20, batch_size=128,shuffle=True)

# Show model performance
train_accuracy = model.evaluate(x_train,y_train)[model.metrics_names.index('accuracy')]
test_accuracy = model.evaluate(x_test,y_test)[model.metrics_names.index('accuracy')]

print("Train/Test accuracy = %.3f/%.3f" % (train_accuracy,test_accuracy))


# Conceptual capacity computation
N = 4000 #Number of points to sample from the training set
         #for the computation of conceptual capacity.
selected_indexes = np.random.permutation(len(x_train))[:N]

Hdf = df_entropy(model,x_train[selected_indexes],verbose=True)

print("Hdf_%d = %.3f" % (N,Hdf))
