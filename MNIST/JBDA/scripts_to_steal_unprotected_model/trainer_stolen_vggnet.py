import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import numpy as np

# Learning rate scheduler
def lr_schedule(epoch):
    learning_rate = 1e-3
    if epoch > 40:
        learning_rate *= 1e-2
    elif epoch > 30:
        learning_rate *= 1e-1
    return learning_rate


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

x_test_adversarial_samples = []
y_test_adversarial_samples = []

with open('X_clean_test_samples.txt','r') as fp:
	lines_x = fp.readlines()
	
with open('Y_clean_test_samples.txt','r') as fp:
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
	x_test_adversarial_samples.append(np.array(list_x_channel, dtype=np.float32))

for line_y in lines_y:
	temp = line_y.strip()
	if(temp != ''):
		y_test_adversarial_samples.append(int(temp))

with open('X_adversarial_samples_0_1.txt','r') as fp:
	lines_x = fp.readlines()
	
with open('Y_adversarial_samples_0_1.txt','r') as fp:
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
	x_test_adversarial_samples.append(np.array(list_x_channel, dtype=np.float32))

for line_y in lines_y:
	temp = line_y.strip()
	if(temp != ''):
		y_test_adversarial_samples.append(int(temp))

with open('X_adversarial_samples_0_2.txt','r') as fp:
	lines_x = fp.readlines()
	
with open('Y_adversarial_samples_0_2.txt','r') as fp:
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
	x_test_adversarial_samples.append(np.array(list_x_channel, dtype=np.float32))

for line_y in lines_y:
	temp = line_y.strip()
	if(temp != ''):
		y_test_adversarial_samples.append(int(temp))

with open('X_adversarial_samples_0_3.txt','r') as fp:
	lines_x = fp.readlines()
	
with open('Y_adversarial_samples_0_3.txt','r') as fp:
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
	x_test_adversarial_samples.append(np.array(list_x_channel, dtype=np.float32))

for line_y in lines_y:
	temp = line_y.strip()
	if(temp != ''):
		y_test_adversarial_samples.append(int(temp))

with open('X_adversarial_samples_0_4.txt','r') as fp:
	lines_x = fp.readlines()
	
with open('Y_adversarial_samples_0_4.txt','r') as fp:
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
	x_test_adversarial_samples.append(np.array(list_x_channel, dtype=np.float32))

for line_y in lines_y:
	temp = line_y.strip()
	if(temp != ''):
		y_test_adversarial_samples.append(int(temp))

with open('X_adversarial_samples_0_5.txt','r') as fp:
	lines_x = fp.readlines()
	
with open('Y_adversarial_samples_0_5.txt','r') as fp:
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
	x_test_adversarial_samples.append(np.array(list_x_channel, dtype=np.float32))

for line_y in lines_y:
	temp = line_y.strip()
	if(temp != ''):
		y_test_adversarial_samples.append(int(temp))





y_test_adversarial_samples = to_categorical(y_test_adversarial_samples, 10)


x_test_adversarial_samples_new = []
for i in range(len(x_test_adversarial_samples)):
	x_test_adversarial_samples_new.append(np.transpose(x_test_adversarial_samples[i],(1, 2, 0)).tolist())


x_mnist = tf.expand_dims(x_mnist_new, axis=-1)
x_test_adversarial_samples = tf.expand_dims(x_test_adversarial_samples_new, axis=-1)

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

lr_scheduler = LearningRateScheduler(lr_schedule)

checkpoint = ModelCheckpoint("stolen_model_base_to_5.h5", monitor='val_loss', verbose = 1, save_best_only = True, mode='min')
callbacks = [checkpoint, lr_scheduler]

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr_schedule(0)), metrics=['accuracy'])

with tf.device('/device:GPU:0'): # /device:GPU:0 for training in GPU
	model.fit(x_test_adversarial_samples, y_test_adversarial_samples, epochs=20, batch_size=64, verbose=1, validation_data=(x_mnist, y_mnist), callbacks=callbacks)

#model_save_path = "/home/seal-lab-workstation/space_submission_2023/Model_Stealing_in_FHE/concrete_cifar_brevitas_training/KnockoffNets/CIFAR100_with_quantized_datapoints/stolen_model.h5"
model = load_model("stolen_model_base_to_5.h5")


print("Checking accuracy of stolen model...")
_, accuracy = model.evaluate(x_mnist, y_mnist)
print(f"Accuracy of stolen model: {np.around(accuracy*100, 2)}")
