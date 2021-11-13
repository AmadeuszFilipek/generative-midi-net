import numpy as np
import tensorflow as tf
import os
import sys
import itertools as it

EPOCHS = 10
BATCH_SIZE = 128
WINDOW_SIZE = 1
INPUT_FILES = [l.rstrip("\r\n") for l in sys.stdin]

def string_to_number_array(s):
   return np.array([int(i) for i in s]).astype(np.float32)


def files_iterator(files):
   for filepath in files:
      if os.path.isfile(filepath):
         yield filepath
      else:
         continue

def iterate_through_file(filepath):
   if not os.path.isfile(filepath):
      raise ValueError("Not a file")
   with open(filepath) as file:
      for line in (l.rstrip("\r\n") for l in file):
         yield line


def steps_stream():
   ''' supports only window=1 for now '''

   for filepath in files_iterator(INPUT_FILES):
      steps_stream = iterate_through_file(filepath)
      previous_line = steps_stream.__next__()
      for line in steps_stream:

         train_x = [string_to_number_array(previous_line)]
         train_x = np.array([train_x])
    
         train_y = string_to_number_array(line)
         train_y = np.array(train_y)

         yield ((train_x), train_y)
         previous_line = line

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=256),
      # input_shape=(1, 128)),
    tf.keras.layers.Dense(256),
    tf.keras.layers.Dense(units=128, activation='relu')
])

def epochify(generator):
   for i in range(EPOCHS):
      INPUT_FILES
      iterator = generator()
      yield iterator

# mean absolute error could also be used
model.compile(
   optimizer='adam',
   loss='mean_squared_error') # tu jeszcze powinna byc funkcja bledu np. mse

history = model.fit(
   # x=dataset, y=None,
   x=steps_stream, y=None,
   epochs=EPOCHS,
   workers=0,
   use_multiprocessing=False,
   # batch_size=BATCH_SIZE,
   # this fraction of train data will be used as validation
   # validation_split=0.1,
   verbose=1,
   # do not shuffle the data after each epoch
   shuffle=False
)



model.summary()

# test_loss = model.evaluate(x_test, y_test)

# print(output)

# def postprocess_output(output):
#    processed_output = []

#    # output[0] because of batch
#    for step in output[0]:
#       processed_output.append(1 if step > 0 else 0)
   
#    return processed_output

# output = model.predict([train[1]])
# print(postprocess_output(output))
