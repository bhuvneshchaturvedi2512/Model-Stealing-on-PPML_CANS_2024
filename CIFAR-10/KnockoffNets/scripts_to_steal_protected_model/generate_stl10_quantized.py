import os
import time
from functools import partial
from pathlib import Path

import torch
from concrete.fhe.compilation.configuration import Configuration
from models import cnv_2w2a
from torch.utils.data import DataLoader
from cifar10_test_quantized import get_train_set_cifar10, get_train_set_stl10

from concrete import fhe
from concrete.ml.deployment.fhe_client_server import FHEModelDev
from concrete.ml.quantization import QuantizedModule
from concrete.ml.torch.compile import compile_brevitas_qat_model
from torch.autograd import grad
import numpy as np
import random

CURRENT_DIR = Path(__file__).resolve().parent
KEYGEN_CACHE_DIR = CURRENT_DIR.joinpath(".keycache")

# Add MPS (for macOS with Apple Silicon or AMD GPUs) support when error is fixed. For now, we
# observe a decrease in torch's top1 accuracy when using MPS devices
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3953
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def getMax(expected_quantized_prediction):
    max_conf = expected_quantized_prediction[0]
    max_label = 0
    for i in range(1,len(expected_quantized_prediction),1):
        if(expected_quantized_prediction[i] > max_conf):
            max_conf = expected_quantized_prediction[i]
            max_label = i
    return max_conf,max_label

def getMin(expected_quantized_prediction):
    min_conf = expected_quantized_prediction[0]
    min_label = 0
    for i in range(1,len(expected_quantized_prediction),1):
        if(expected_quantized_prediction[i] < min_conf):
            min_conf = expected_quantized_prediction[i]
            min_label = i
    return min_conf,min_label

def measure_execution_time(func):
    """Run a function and return execution time and outputs.

    Usage:
        def f(x, y):
            return x + y
        output, execution_time = measure_execution_time(f)(x,y)
    """

    def wrapper(*args, **kwargs):
        # Get the current time
        start = time.time()

        # Call the function
        result = func(*args, **kwargs)

        # Get the current time again
        end = time.time()

        # Calculate the execution time
        execution_time = end - start

        # Return the result and the execution time
        return result, execution_time

    return wrapper


# Instantiate the model
torch_model = cnv_2w2a(pre_trained=False)
torch_model.eval()


# Load the saved parameters using the available checkpoint
checkpoint = torch.load(
    CURRENT_DIR.joinpath("experiments/CNV_2W2A_2W2A_20231101_103516/checkpoints/best.tar"),
    map_location=DEVICE,
)
torch_model.load_state_dict(checkpoint["state_dict"], strict=False)

# Import and load the CIFAR test dataset
train_set = get_train_set_cifar10(dataset="CIFAR10", datadir=CURRENT_DIR.joinpath(".datasets/"))
train_loader = DataLoader(train_set, batch_size=100, shuffle=False)

# Get the first sample
samples_data = []
count = 0

for i, data in enumerate(train_loader):
    samples_data.append(data[0])
    samples_data.append(data[1])
    count = count + 1
    if(count == 1):
        break
x, y = samples_data

# Import and load the CIFAR test dataset
train_set_stl10 = get_train_set_stl10(dataset="STL10", datadir=CURRENT_DIR.joinpath(".datasets/"))
train_loader_stl10 = DataLoader(train_set_stl10, batch_size=50000, shuffle=False)

# Get the first sample
samples_data = []
count = 0

for i, data in enumerate(train_loader_stl10):
    samples_data.append(data[0])
    samples_data.append(data[1])
    count = count + 1
    if(count == 1):
        break
x_stl10, y_stl10 = samples_data

NUM_SAMPLES = len(x_stl10)
print("Length: " + str(NUM_SAMPLES))



# Parameter `enable_unsafe_features` and `use_insecure_key_cache` are needed in order to be able to
# cache generated keys through `insecure_key_cache_location`. As the name suggests, these
# parameters are unsafe and should only be used for debugging in development
# Multi-parameter strategy is used in order to speed-up the FHE executions
configuration = Configuration(
    dump_artifacts_on_unexpected_failures=False,
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location=KEYGEN_CACHE_DIR,
)

print("Compiling the model.")
quantized_numpy_module, compilation_execution_time = measure_execution_time(
    compile_brevitas_qat_model
)(torch_model, x, configuration=configuration, rounding_threshold_bits=6, p_error=0.01)
assert isinstance(quantized_numpy_module, QuantizedModule)

print(f"Compilation time took {compilation_execution_time} seconds")

# Display the max bit-width in the model
print(
    "Max bit-width used in the circuit: ",
    f"{quantized_numpy_module.fhe_circuit.graph.maximum_integer_bit_width()} bits",
)

# Save the graph and mlir
#print("Saving graph and mlir to disk.")
#open("cifar10.graph", "w").write(str(quantized_numpy_module.fhe_circuit))
#open("cifar10.mlir", "w").write(quantized_numpy_module.fhe_circuit.mlir)

# Data torch to numpy
x_numpy = x_stl10.numpy()

r = 3
M = 10
count = 0
Lambda = 180

gap_comp = 0

for image_index in range(NUM_SAMPLES):
    test_x_numpy = x_numpy[image_index : image_index + 1]
    
    q_x_numpy, quantization_execution_time = measure_execution_time(
        quantized_numpy_module.quantize_input
    )(test_x_numpy)
    
    for i in range(len(q_x_numpy[0])):
        for j in range(len(q_x_numpy[0][0])):
            for k in range(len(q_x_numpy[0][0][0])):
                with open('X_stl10_test.txt','a') as fp:
                    fp.write(str(test_x_numpy[0][i][j][k]) + ",")
            with open('X_stl10_test.txt','a') as fp:
                fp.write("#")
        with open('X_stl10_test.txt','a') as fp:
            fp.write("|")
    with open('X_stl10_test.txt','a') as fp:
        fp.write("\n")
    
    p_error = quantized_numpy_module.fhe_circuit.p_error
    expected_quantized_prediction, clear_inference_time = measure_execution_time(
        partial(quantized_numpy_module.fhe_circuit.graph, p_error=p_error)
    )(q_x_numpy)

    max_conf,max_label = getMax(expected_quantized_prediction[0])
    min_conf,min_label = getMin(expected_quantized_prediction[0])
    
    predicted_label = max_label
    
    gap = max_conf - min_conf
    
    if(max_conf < Lambda):
    	count = count + 1
    
    	if(count % M <= r and gap > gap_comp):
        	gap_comp = gap
        
    	elif(count % M > r and gap > gap_comp):
        	predicted_label = min_label
        
    	elif(count % M == 0):
        	predicted_label = min_label
        	gap_comp = 0
        	count = 0
    
    with open('Y_stl10_test.txt','a') as fp:
        fp.write(str(predicted_label) + '\n')  
    
    if(image_index % 100 == 0):
        print("First " + str(image_index) + " samples written")
  
print("Samples generated")
