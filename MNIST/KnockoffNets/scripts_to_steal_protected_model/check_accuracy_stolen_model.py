import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import numpy as np

x_mnist = []
y_mnist = []

with open('X_mnist_test.txt','r') as fp:
	lines_x = fp.readlines()
	
with open('Y_mnist_test.txt','r') as fp:
	lines_y = fp.readlines()
	

for line_x in lines_x:
	list_x_channel = []
	temp_row = line_x.strip().split('#')
	list_x_row = []
	for row in temp_row:
		if(row != ''):
			temp_column = row.strip().split(',')
			list_x_column = []
			for column in temp_column:
				if(column != ''):
					list_x_column.append(float(column))
			list_x_row.append(np.array(list_x_column, dtype=np.float32))
	list_x_channel.append(list(list_x_row))
	x_mnist.append(np.array(list_x_channel, dtype=np.float32))

for line_y in lines_y:
	temp = line_y.strip()
	if(temp != ''):
		y_mnist.append(int(temp))

y_mnist = to_categorical(y_mnist, 10)

x_mnist_new = []
for i in range(len(lines_y)):
	x_mnist_new.append(np.transpose(x_mnist[i],(1, 2, 0)).tolist())

x_mnist = tf.expand_dims(x_mnist_new, axis=-1)

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model = load_model("stolen_model.h5")


print("Checking accuracy of stolen model...")
_, accuracy = model.evaluate(x_mnist, y_mnist)
print(f"Accuracy of stolen model: {np.around(accuracy*100, 2)}")
