import matplotlib.pyplot as plt
import numpy as np
import torch

# Concrete-Python
from concrete.fhe.compilation import Configuration

# The QAT model
from model import MNISTQATModel  # pylint: disable=no-name-in-module
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from tqdm import tqdm

# Concrete ML
from concrete.ml.torch.compile import compile_brevitas_qat_model

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
    
def manage_dataset_mnist(train_kwargs, test_kwargs):
    """Get training and test parts of MNIST data-set."""

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(torch.flatten),
        ]
    )
    
    root_path = "./data"

    # Manage data-sets
    dataset1 = datasets.MNIST(root=root_path, download=False, train=True, transform=transform)
    dataset2 = datasets.MNIST(root=root_path, download=False, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader
    
def manage_dataset_fmnist(train_kwargs, test_kwargs):
    """Get training and test parts of MNIST data-set."""

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3205,)),
            transforms.Lambda(torch.flatten),
        ]
    )
    
    root_path = "./data"

    # Manage data-sets
    dataset1 = datasets.FashionMNIST(root=root_path, download=True, train=True, transform=transform)
    dataset2 = datasets.FashionMNIST(root=root_path, download=True, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader

def compile_and_test(
    model,
    use_simulation,
    compilation_data,
    test_data,
    test_data_length,
    test_target,
    show_mlir,
    current_index,
):
    # Compile the QAT model and test
    configuration = Configuration(
        enable_unsafe_features=True,  # This is for our tests only, never use that in prod
        use_insecure_key_cache=True,  # This is for our tests only, never use that in prod
        insecure_key_cache_location="/tmp/keycache",
    )

    if use_simulation:
        print(f"\n{current_index}. Compiling with the FHE simulation")
    else:
        print(f"\n{current_index}. Compiling in FHE")

    q_module = compile_brevitas_qat_model(
        model,
        compilation_data,
        configuration=configuration,
        show_mlir=show_mlir,
    )

    # Check max bit-width
    max_bit_width = q_module.fhe_circuit.graph.maximum_integer_bit_width()

    if max_bit_width > 8:
        raise Exception(
            f"Too large bit-width ({max_bit_width}): training this network resulted in an "
            "accumulator size that is too large. Possible solutions are:"
            "    - this network should, on average, have 8bit accumulators. In your case an unlucky"
            f"initialization resulted in {max_bit_width} accumulators. You can try to train the "
            "network again"
            "    - reduce the sparsity to reduce the number of active neuron connections"
            "    - if the weight and activation bit-width is more than 2, you can try to reduce one"
            "or both to a lower value"
        )

    # Check the accuracy
    if use_simulation:
        print(
            f"\n{current_index + 1}. Checking accuracy with the FHE simulation mode "
            f"(length {test_data_length})"
        )
    else:
        print(f"\n{current_index + 1}. Checking accuracy in FHE (length {test_data_length})")

    # Key generation
    if not use_simulation:
        q_module.fhe_circuit.keygen()

    correct_fhe = 0

    # Reduce the test data, since very slow in FHE
    reduced_test_data = test_data[0:test_data_length, :]
    test_target = test_target[0:test_data_length, :]

    fhe_mode = "simulate" if use_simulation else "execute"

    for image_index in range(test_data_length):
        #print(len(reduced_test_data[image_index]))
        #print(reduced_test_data[image_index])
        query = []
        query.append(reduced_test_data[image_index])
        prediction = q_module.forward(query, fhe=fhe_mode)
        #print(prediction)
        max_conf,max_label = getMax(prediction[0])
        count = 0
        for i in range(len(reduced_test_data[image_index])):
            count = count + 1
            with open('X_fmnist_test.txt','a') as fp:
                fp.write(str(reduced_test_data[image_index][i]))
                if(count != 28):
                    fp.write(",")
                else:
                    fp.write("#")
                    count = 0
        with open('X_fmnist_test.txt','a') as fp:
            fp.write("\n")
        with open('Y_fmnist_test.txt','a') as fp:
            fp.write(str(max_label) + '\n')
        if(image_index % 100 == 0):
            print("First " + str(image_index) + " samples written")
        #exit(0)

    #correct_fhe = (np.argmax(prediction, axis=1) == test_target.ravel()).sum()

    
# Options: the most important ones
sparsity = 4
a_bits = 2
w_bits = 2
test_data_length_full = 50000
show_mlir = False
batch_size = 32
test_batch_size = 32
use_cuda_if_available = True

# Training and test arguments
train_kwargs = {"batch_size": batch_size}
test_kwargs = {"batch_size": test_batch_size}

# Cuda management
use_cuda = torch.cuda.is_available() and use_cuda_if_available
device = torch.device("cuda" if use_cuda else "cpu")

if use_cuda:
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

print(f"\nUsing {device} device\n")

# Manage data-set
train_loader_mnist, test_loader_mnist = manage_dataset_mnist(train_kwargs, test_kwargs)

train_loader_fmnist, test_loader_fmnist = manage_dataset_fmnist(train_kwargs, test_kwargs)

img_size = train_loader_mnist.dataset.data[0].shape[0]

# Model definition
model = MNISTQATModel(a_bits, w_bits)
checkpoint = torch.load("state_dict.pt", map_location=device)
model.load_state_dict(checkpoint)

# Training part
print(
    f"Performing MNIST task with {a_bits}-bits in activation quantization, {w_bits}-bits in weight quantization and a "
    f"sparsity of {sparsity}."
)

# Prepare compilation data
compilation_data = np.zeros((len(train_loader_mnist.dataset), img_size * img_size))
idx = 0

for data, target in tqdm(train_loader_mnist):
    for idx_batch, im in enumerate(data.numpy()):
        compilation_data[idx] = im
        idx += 1
   
# Prepare MNIST tests
test_data = np.zeros((len(test_loader_mnist.dataset), img_size * img_size))
test_target = np.zeros((len(test_loader_mnist.dataset), 1))
idx = 0

for data, target in tqdm(test_loader_mnist):
    target_np = target.cpu().numpy()
    for idx_batch, im in enumerate(data.numpy()):
        test_data[idx] = im
        test_target[idx] = target_np[idx_batch]
        idx += 1
   
# Prepare FashionMNIST tests
test_data = np.zeros((len(train_loader_fmnist.dataset), img_size * img_size))
test_target = np.zeros((len(train_loader_fmnist.dataset), 1))
idx = 0

for data, target in tqdm(train_loader_fmnist):
    target_np = target.cpu().numpy()
    for idx_batch, im in enumerate(data.numpy()):
        test_data[idx] = im
        test_target[idx] = target_np[idx_batch]
        idx += 1
        
# Test in the FHE simulation and in real FHE computation
accuracy = {}
current_index = 3

use_simulation = True

compile_and_test(
    model.cpu(),
    use_simulation,
    compilation_data,
    test_data,
    test_data_length_full,
    test_target,
    show_mlir,
    current_index,
)

        

