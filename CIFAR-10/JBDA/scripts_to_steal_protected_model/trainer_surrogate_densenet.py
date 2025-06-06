import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy as np

# Learning rate scheduler
def lr_schedule(epoch):
    learning_rate = 1e-3
    if epoch > 40:
        learning_rate *= 1e-2
    elif epoch > 30:
        learning_rate *= 1e-1
    return learning_rate


x_cifar10 = []
y_cifar10 = []

with open('X_cifar10_test.txt','r') as fp:
	lines_x = fp.readlines()
	
with open('Y_cifar10_test.txt','r') as fp:
	lines_y = fp.readlines()
	

for line_x in lines_x:
	temp_channel = line_x.strip().split('|')
	list_x_channel = []
	for channel in temp_channel:
		if(channel != ''):
			temp_row = channel.strip().split('#')
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
	x_cifar10.append(np.array(list_x_channel, dtype=np.float32))

for line_y in lines_y:
	temp = line_y.strip()
	if(temp != ''):
		y_cifar10.append(int(temp))

y_cifar10 = to_categorical(y_cifar10, 10)

x_cifar10_new = []
for i in range(len(lines_y)):
	x_cifar10_new.append(np.transpose(x_cifar10[i],(1, 2, 0)).tolist())



x_test_adversarial_samples = []
y_test_adversarial_samples = []

with open('X_clean_test_samples.txt','r') as fp:
	lines_x = fp.readlines()
	
with open('Y_clean_test_samples.txt','r') as fp:
	lines_y = fp.readlines()

for line_x in lines_x:
	temp_channel = line_x.strip().split('|')
	list_x_channel = []
	for channel in temp_channel:
		if(channel != ''):
			temp_row = channel.strip().split('#')
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
	temp_row = line_x.strip().split('|')
	list_x_row = []
	for row in temp_row:
		if(row != ''):
			temp_column = row.strip().split('#')
			list_x_column = []
			for column in temp_column:
				if(column != ''):
					temp_channel = column.strip().split(',')
					list_x_channel = []
					for channel in temp_channel:
						if(channel != ''):
							list_x_channel.append(float(channel))
					list_x_column.append(np.array(list_x_channel, dtype=np.float32))
			list_x_row.append(list(list_x_column))
	x_test_adversarial_samples.append(np.array(list_x_row, dtype=np.float32))
	
for line_y in lines_y:
	temp = line_y.strip()
	if(temp != ''):
		y_test_adversarial_samples.append(int(temp))

with open('X_adversarial_samples_0_2.txt','r') as fp:
	lines_x = fp.readlines()
	
with open('Y_adversarial_samples_0_2.txt','r') as fp:
	lines_y = fp.readlines()
	
for line_x in lines_x:
	temp_row = line_x.strip().split('|')
	list_x_row = []
	for row in temp_row:
		if(row != ''):
			temp_column = row.strip().split('#')
			list_x_column = []
			for column in temp_column:
				if(column != ''):
					temp_channel = column.strip().split(',')
					list_x_channel = []
					for channel in temp_channel:
						if(channel != ''):
							list_x_channel.append(float(channel))
					list_x_column.append(np.array(list_x_channel, dtype=np.float32))
			list_x_row.append(list(list_x_column))
	x_test_adversarial_samples.append(np.array(list_x_row, dtype=np.float32))
	
for line_y in lines_y:
	temp = line_y.strip()
	if(temp != ''):
		y_test_adversarial_samples.append(int(temp))

with open('X_adversarial_samples_0_3.txt','r') as fp:
	lines_x = fp.readlines()
	
with open('Y_adversarial_samples_0_3.txt','r') as fp:
	lines_y = fp.readlines()
	
for line_x in lines_x:
	temp_row = line_x.strip().split('|')
	list_x_row = []
	for row in temp_row:
		if(row != ''):
			temp_column = row.strip().split('#')
			list_x_column = []
			for column in temp_column:
				if(column != ''):
					temp_channel = column.strip().split(',')
					list_x_channel = []
					for channel in temp_channel:
						if(channel != ''):
							list_x_channel.append(float(channel))
					list_x_column.append(np.array(list_x_channel, dtype=np.float32))
			list_x_row.append(list(list_x_column))
	x_test_adversarial_samples.append(np.array(list_x_row, dtype=np.float32))
	
for line_y in lines_y:
	temp = line_y.strip()
	if(temp != ''):
		y_test_adversarial_samples.append(int(temp))

with open('X_adversarial_samples_0_4.txt','r') as fp:
	lines_x = fp.readlines()
	
with open('Y_adversarial_samples_0_4.txt','r') as fp:
	lines_y = fp.readlines()
	
for line_x in lines_x:
	temp_row = line_x.strip().split('|')
	list_x_row = []
	for row in temp_row:
		if(row != ''):
			temp_column = row.strip().split('#')
			list_x_column = []
			for column in temp_column:
				if(column != ''):
					temp_channel = column.strip().split(',')
					list_x_channel = []
					for channel in temp_channel:
						if(channel != ''):
							list_x_channel.append(float(channel))
					list_x_column.append(np.array(list_x_channel, dtype=np.float32))
			list_x_row.append(list(list_x_column))
	x_test_adversarial_samples.append(np.array(list_x_row, dtype=np.float32))
	
for line_y in lines_y:
	temp = line_y.strip()
	if(temp != ''):
		y_test_adversarial_samples.append(int(temp))

with open('X_adversarial_samples_0_5.txt','r') as fp:
	lines_x = fp.readlines()
	
with open('Y_adversarial_samples_0_5.txt','r') as fp:
	lines_y = fp.readlines()
	
for line_x in lines_x:
	temp_row = line_x.strip().split('|')
	list_x_row = []
	for row in temp_row:
		if(row != ''):
			temp_column = row.strip().split('#')
			list_x_column = []
			for column in temp_column:
				if(column != ''):
					temp_channel = column.strip().split(',')
					list_x_channel = []
					for channel in temp_channel:
						if(channel != ''):
							list_x_channel.append(float(channel))
					list_x_column.append(np.array(list_x_channel, dtype=np.float32))
			list_x_row.append(list(list_x_column))
	x_test_adversarial_samples.append(np.array(list_x_row, dtype=np.float32))
	
for line_y in lines_y:
	temp = line_y.strip()
	if(temp != ''):
		y_test_adversarial_samples.append(int(temp))

y_test_adversarial_samples = to_categorical(y_test_adversarial_samples, 10)

x_test_adversarial_new = []
for i in range(len(x_test_adversarial_samples)):
	x_test_adversarial_new.append(np.transpose(x_test_adversarial_samples[i],(1, 2, 0)).tolist())

x_cifar10_new = np.array(x_cifar10_new)
y_cifar10 = np.array(y_cifar10)


x_test_adversarial_new = np.array(x_test_adversarial_new)
y_test_adversarial_samples = np.array(y_test_adversarial_samples)



base_model = DenseNet121(weights='imagenet', include_top=False)

# Add custom layers on top of DenseNet
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# Create final model
model = Model(inputs=base_model.input, outputs=predictions)

lr_scheduler = LearningRateScheduler(lr_schedule)

checkpoint = ModelCheckpoint("stolen_model_base_to_5.h5", monitor='val_loss', verbose = 1, save_best_only = True, mode='min')
callbacks = [checkpoint, lr_scheduler]

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr_schedule(0)), metrics=['accuracy'])

with tf.device('/device:GPU:0'): # /device:GPU:0 for training in GPU
	model.fit(x_test_adversarial_new, y_test_adversarial_samples, epochs=50, batch_size=64, verbose=1, validation_data=(x_cifar10_new, y_cifar10), callbacks=callbacks)

model = load_model("stolen_model_base_to_5.h5")

_, accuracy = model.evaluate(x_cifar10_new, y_cifar10)
print(f"Accuracy of stolen model: {np.around(accuracy*100, 2)}")
