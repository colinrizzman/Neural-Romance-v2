# github.com/colinrizzman
# pip install numpy tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
import signal
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
from time import time_ns
from os.path import isfile
from os.path import isdir
from os import mkdir
from pathlib import Path
from datetime import datetime
from fractions import Fraction

# print everything / no truncations
np.set_printoptions(threshold=sys.maxsize)

# hyperparameters
optimiser = 'adam'
activator = 'relu'
inputsize = 26
outputsize = 1
layers = 6
layer_units = 32 # 32, 256
batches = 512 # 24, 512
epoches = 33333 # 12500, 33333
topo = 1
earlystop = 3000 # 3000, 0 = off, anything above is the patience value

# load hyperparameters
argc = len(sys.argv)
if argc >= 2: layers = int(sys.argv[1])
if argc >= 3: layer_units = int(sys.argv[2])
if argc >= 4: batches = int(sys.argv[3])
if argc >= 5: epoches = int(sys.argv[4])
if argc >= 6: activator = sys.argv[5]
if argc >= 7: optimiser = sys.argv[6]
if argc >= 8: topo = int(sys.argv[7])
if argc >= 9: earlystop = int(sys.argv[8])

# print hyperparameters
print("\n--Hyperparameters")
print("layers:", layers)
print("layer_units:", layer_units)
print("batches:", batches)
print("epoches:", epoches)
print("activator:", activator)
print("optimiser:", optimiser)
print("topo:", topo)
print("earlystop:", earlystop)

# model name
model_name = 'models/' + activator + '_' + optimiser + '_' + str(layers) + '_' + str(layer_units) + '_' + str(batches) + '_' + str(epoches) + '_' + str(topo)

##########################################
#   HELPERS
##########################################
class PrintFullLoss(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        numeric = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        parts = [f"{k}: {v:.10f}" for k, v in numeric.items()]
        print(f" - " + " - ".join(parts))
print_loss = PrintFullLoss()

early_stop = EarlyStopping(
    monitor='loss',
    patience=earlystop,
    min_delta=1e-7,
    verbose=1,
    mode='min'
)

class _SigintFlag:
    def __init__(self): self.stop = False
    def __call__(self, signum, frame): self.stop = True
sigint_flag = _SigintFlag()
signal.signal(signal.SIGINT, sigint_flag)

class GracefulStop(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        if sigint_flag.stop: self.model.stop_training = True
    def on_epoch_end(self, epoch, logs=None):
        if sigint_flag.stop: self.model.stop_training = True

##########################################
#   LOAD
##########################################
print("\n--Loading Dataset")
st = time_ns()

dataset_size = sum(1 for _ in Path('training_data.txt').open())
print("Dataset Size:", "{:,}".format(dataset_size))

if isfile("train_x.npy"):
    train_x = np.load("train_x.npy")
    train_y = np.load("train_y.npy")
else:
    data = np.loadtxt('training_data.txt')
    train_x = data[:, :inputsize] # first columns
    train_y = data[:, inputsize]  # last column
    np.save("train_x.npy", train_x)
    np.save("train_y.npy", train_y)

# print(train_x.shape)
# print(train_x)
# print(train_y.shape)
# print(train_y)
# sys.exit()

timetaken = (time_ns()-st)/1e+9
print("Time Taken:", "{:.2f}".format(timetaken), "seconds")

##########################################
#   TRAIN
##########################################
print("\n--Model Topology")

# construct neural network
model = Sequential()
model.add(Input(shape=(inputsize,)))
model.add(Dense(layer_units, activation=activator))
if topo == 0:
    for x in range(layers): model.add(Dense(layer_units, activation=activator))
elif topo == 1:
    for x in range(layers-1): model.add(Dense(layer_units, activation=activator))
    model.add(Dense(int(layer_units/2), activation=activator))
elif topo == 2:
    dunits = layer_units
    for x in range(layers):
        model.add(Dense(int(dunits), activation=activator))
        dunits=dunits/2
model.add(Dense(outputsize))

# output summary
model.summary()

# set optimiser
if optimiser == 'adam':
    optim = keras.optimizers.Adam(learning_rate=0.001)
elif optimiser == 'sgd':
    optim = keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
elif optimiser == 'sgd-decay':
    decay = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.3, decay_steps=epoches*dataset_size, decay_rate=0.1)
    optim = keras.optimizers.SGD(learning_rate=decay, momentum=0.0, nesterov=False)
elif optimiser == 'sgd-decay-cosine':
    decay = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.3, decay_steps=epoches*dataset_size, alpha=0.1)
    optim = keras.optimizers.SGD(learning_rate=decay, momentum=0.0, nesterov=False)
elif optimiser == 'sgd-decay-fine':
    decay = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.1, decay_steps=epoches*dataset_size, decay_rate=0.01)
    optim = keras.optimizers.SGD(learning_rate=decay, momentum=0.0, nesterov=False)
elif optimiser == 'momentum':
    optim = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False)
elif optimiser == 'nesterov':
    optim = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
elif optimiser == 'nadam':
    optim = keras.optimizers.Nadam(learning_rate=0.001)
elif optimiser == 'adagrad':
    optim = keras.optimizers.Adagrad(learning_rate=0.001)
elif optimiser == 'rmsprop':
    optim = keras.optimizers.RMSprop(learning_rate=0.001)
elif optimiser == 'adadelta':
    optim = keras.optimizers.Adadelta(learning_rate=0.001)
elif optimiser == 'adamax':
    optim = keras.optimizers.Adamax(learning_rate=0.001)
elif optimiser == 'ftrl':
    optim = keras.optimizers.Ftrl(learning_rate=0.001)

# compile & train
print("\n--Training Model")
model.compile(optimizer=optim, loss='mean_squared_error')
st = time_ns()
if earlystop == 0:  history = model.fit(train_x, train_y, epochs=epoches, batch_size=batches, callbacks=[GracefulStop(), print_loss])
else:               history = model.fit(train_x, train_y, epochs=epoches, batch_size=batches, callbacks=[early_stop, GracefulStop(), print_loss])
model_name = model_name + "_" + "L[{:.6f}]".format(history.history['loss'][-1])
timetaken = (time_ns()-st)/1e+9
print("\nTime Taken:", "{:.2f}".format(timetaken), "seconds")

##########################################
#   EXPORT
##########################################
print("\n--Exporting Model")
st = time_ns()

# predict
p1 = model.predict(np.array([[float(Fraction(4, 9))] * inputsize], dtype=np.float32), verbose=0)
p2 = model.predict(np.array([[0.0] * inputsize], dtype=np.float32), verbose=0)
p3 = model.predict(np.array([[1.0] * inputsize], dtype=np.float32), verbose=0)

# save weights for JS array
if not isdir('models'): mkdir('models')
li = 0
f = open(model_name + ".txt", "w")
f.write("// loss: " + "{:.12f}".format(history.history['loss'][-1]) + "\n")
f.write("// Reset Percentage: " + "{:.2f}".format(p1[0][0]*100) + "%\n")
f.write("// Min Percentage: " + "{:.2f}".format(p2[0][0]*100) + "%\n")
f.write("// Max Percentage: " + "{:.2f}".format(p3[0][0]*100) + "%\n\n")
if f:
    for layer in model.layers:
        total_layer_weights = layer.get_weights()[0].transpose().flatten().shape[0]
        total_layer_units = layer.units
        layer_weights_per_unit = total_layer_weights / total_layer_units

        f.write("const L" + str(li) + " = new Float32Array([")
        isfirst = 0
        wc = 0
        bc = 0
        if layer.get_weights() != []:
            for weight in layer.get_weights()[0].transpose().flatten():
                wc += 1
                if isfirst == 0:
                    f.write(str(weight))
                    isfirst = 1
                else:
                    f.write("," + str(weight))
                if wc == layer_weights_per_unit:
                    f.write("," + str(layer.get_weights()[1].transpose().flatten()[bc]))
                    wc = 0
                    bc += 1
        f.write("])\n")
        li += 1
f.close()

timetaken = (time_ns()-st)/1e+9
print("\nTime Taken:", "{:.2f}".format(timetaken), "seconds\n")

# final
print("--Finalization\n")
print("Reset Percentage: " + "{:.2f}".format(p1[0][0]*100) + "%")
print("Min   Percentage: " + "{:.2f}".format(p2[0][0]*100) + "%")
print("Max   Percentage: " + "{:.2f}".format(p3[0][0]*100) + "%\n")
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": " + model_name + "\n")
