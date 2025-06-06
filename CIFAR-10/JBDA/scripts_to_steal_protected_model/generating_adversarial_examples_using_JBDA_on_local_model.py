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

def jacobian_matrix(model, input_data, label_index):
    with tf.GradientTape(persistent=True) as tape:
        # Watch the input tensor
        tape.watch(input_data)
        
        # Get the model prediction
        predictions = model(input_data)
        
        # Extract the specific label's prediction
        target_prediction = predictions[:, label_index]
    
    # Compute the Jacobian matrix
    jacobian_matrix = tape.jacobian(target_prediction, input_data)

    return jacobian_matrix


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

x_test_adversarial_new = []
for i in range(len(lines_y)):
	x_test_adversarial_new.append(np.transpose(x_test_adversarial_samples[i],(1, 2, 0)).tolist())
	
n_samples = 1000

# Load the pre-trained model
print("Loading unprotected model...")
model = load_model('stolen_model_base_to_4.h5')

x = tf.cast(x_test_adversarial_new, tf.float32)

eps = 128/255
adv_x = []

for i in range(int(n_samples/2),n_samples,1): #0,int(n_samples/2),1 for first 500 #int(n_samples/2),n_samples,1 for next 500
    x_i = []
    x_arr = []
    x_i.append(tf.convert_to_tensor(np.array(x[i], dtype=np.float32).reshape(1,32,32,3)))
    jacobian_matrix_result = jacobian_matrix(model, x_i, y_test_adversarial_samples[i])
    jacobian_sign = tf.sign(jacobian_matrix_result)
    x_arr.append(np.array(x_i, dtype=np.float32))
    x_arr[0] = x_arr[0] + eps*jacobian_sign
    adv_x.append(np.array(x_arr[0][0], dtype=np.float32))
    if(i % 50 == 49):
        print("Processed first " + str(i + 1) + " samples")
adv_x_new = []
for i in range(0,int(n_samples/2),1):
	adv_x_new.append(np.transpose(adv_x[i][0][0],(2, 0, 1)).tolist())

for i in range(0,int(n_samples/2),1):
	for j in range(len(adv_x_new[0])):
		for k in range(len(adv_x_new[0][0])):
			for l in range(len(adv_x_new[0][0][0])):
				with open('X_adversarial_samples_0_5.txt','a') as fp:
					fp.write(str(adv_x_new[i][j][k][l]) + ",")
			with open('X_adversarial_samples_0_5.txt','a') as fp:
				fp.write("#")
		with open('X_adversarial_samples_0_5.txt','a') as fp:
			fp.write("|")
	with open('X_adversarial_samples_0_5.txt','a') as fp:
		fp.write("\n")

print("Samples with eps=" + str(eps) + " generated")
