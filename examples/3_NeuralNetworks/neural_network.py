# coding:utf8
""" Neural Network.

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).

This example is using TensorFlow layers, see 'neural_network_raw' example for
a raw implementation with variables.

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function
import helper

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description="""A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).""")

parser.add_argument("--learning_rate",type=float, default=0.1, help="model learning rate")
parser.add_argument("--num_steps",type=int, default=1000, help="model num of step")
parser.add_argument("--batch_size",type=int, default=128, help="model batch size")
parser.add_argument("--display_step",type=int,default=100,help="model display step")

parser.add_argument("--n_hidden_1",type=int,default=256,help="1st layer number of neurons")
parser.add_argument("--n_hidden_2",type=int,default=256,help="2nd layer number of neurons")
parser.add_argument("--num_input",type=int,default=784,help="MNIST data input (img shape: 28*28)")
parser.add_argument("--num_classes",type=int,default=10,help="MNIST total classes (0-9 digits)")

parser.add_argument("--input_data",type=str,default="/tmp/data/", help="model input data dir")
parser.add_argument("--model_name",type=lambda s: unicode(s,'utf8').strip(u'模型'),default="test", help="model name")

args = parser.parse_args()

learning_rate = args.learning_rate
num_steps = args.num_steps
batch_size = args.batch_size
display_step = args.display_step

n_hidden_1 = args.n_hidden_1 # 1st layer number of neurons
n_hidden_2 = args.n_hidden_2 # 2nd layer number of neurons
num_input = args.num_input # MNIST data input (img shape: 28*28)
num_classes = args.num_classes # MNIST total classes (0-9 digits)

model_dir = helper.model_dir(__file__, args.model_name)

mnist = input_data.read_data_sets(args.input_data, one_hot=False)
# Parameters
# learning_rate = 0.1
# num_steps = 1000
# batch_size = 128
# display_step = 100

# Network Parameters
# n_hidden_1 = 256 # 1st layer number of neurons
# n_hidden_2 = 256 # 2nd layer number of neurons
# num_input = 784 # MNIST data input (img shape: 28*28)
# num_classes = 10 # MNIST total classes (0-9 digits)


# Define the neural network
def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['images']
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    logits = neural_net(features)

    # Predictions
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

# Build the Estimator
model = tf.estimator.Estimator(model_fn, model_dir=model_dir)

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# Train the Model
model.train(input_fn, steps=num_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])


helper.start_tensorboard()
