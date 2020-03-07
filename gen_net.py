import numpy as np
import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 128
WINDOW_SIZE = 1

raw_input = [
   '00000000000000000000000000000000000000000000000000000000000000000000000010000000000010000000000000000000000000000000000000000000',
   '00000000000000000000000000000000000000000000000000000000000000000000100000000000100000000000000000000000000000000000000000000000',
   '00000000000000000000000000000000000000000000000000000000000000000001000000000001000000000000000000000000000000000000000000000000',
   '00000000000000000000000000000000000000000000000000000000000000000001000000000001000000000000000000000000000000000000000000000000',
   '00000000000000000000000000000000000000000000000000000000000000000000100000000000100000000000000000000000000000000000000000000000',
   '00000000000000000000000000000000000000000000000000000000000000000100000000000100000000000000000000000000000000000000000000000000'
]

string_to_number_array = lambda s: [int(i) for i in s]
bits_input = [[string_to_number_array(string)] for string in raw_input]

train = [[string_to_number_array(string)] for string in raw_input[0:-1]] # whole table without last element
labels = [string_to_number_array(string) for string in raw_input[1:]] # same table but without first element
# so that the indexes in both tables are shifted by one

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(
      units=256,
      input_shape=(1, 128)),   # nie musze podawac, bo sobie z danych uczacych wyciagnie
    tf.keras.layers.Dense(256),
    tf.keras.layers.Dense(128) # output
])

# mean absolute error could also be used
model.compile(
   optimizer='adam',
   loss='mean_squared_error') # tu jeszcze powinna byc funkcja bledu np. mse

history = model.fit(
   np.array(train), np.array(labels), 
   epochs=EPOCHS,
   batch_size=BATCH_SIZE,
   # this fraction of train data will be used as validation
   validation_split=0.1,
   verbose=1,
   # do not shuffle the data after each epoch
   shuffle=False
)

model.summary()

# test_loss = model.evaluate(x_test, y_test)


output = model.predict([train[1]]) # ciuuf ciuuf net-hype-train
print(output)