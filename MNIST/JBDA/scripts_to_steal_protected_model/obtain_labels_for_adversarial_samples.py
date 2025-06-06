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
    
def manage_dataset(train_kwargs, test_kwargs):
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

def compile_and_test(
    model,
    use_simulation,
    compilation_data,
    test_data_length,
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
        
    #augmenting adversarial samples with perturbation 0.1
    x_test_adversarial_samples = []
    y_test_adversarial_samples = []
        
    with open('X_adversarial_samples_0_5.txt','r') as fp:
        lines_x = fp.readlines()
   
    for line_x in lines_x:
        temp_channel = line_x.strip().split('|')
        list_x_channel = []
        for channel in temp_channel:
            if(channel != ''):
                temp_row = channel.strip().split('#')
                for row in temp_row:
                    if(row != ''):
                        temp_column = row.strip().split(',')
                        for column in temp_column:
                            if(column != ''):
                                list_x_channel.append(float(column))
        x_test_adversarial_samples.append(list(list_x_channel))

    fhe_mode = "simulate" if use_simulation else "execute"

    r = 3
    M = 10
    count = 0
    Lambda = 5

    gap_comp = 0

    for image_index in range(test_data_length):
        query = []
        query.append(x_test_adversarial_samples[image_index])
        prediction = q_module.forward(query, fhe=fhe_mode)
        
        max_conf,max_label = getMax(prediction[0])
        
        max_conf,max_label = getMax(prediction[0])
        min_conf,min_label = getMin(prediction[0])
        
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
        
        with open('Y_adversarial_samples_0_5.txt','a') as fp:
            fp.write(str(predicted_label) + '\n')
        if(image_index % 100 == 0):
            print("First " + str(image_index) + " samples written")
    
# Options: the most important ones
epochs = 100
sparsity = 4
a_bits = 2
w_bits = 2
do_training = True
save_model = True

# Options: can be changed
lr = 0.02
gamma = 0.33
test_data_length_reduced = 2 # This is notably the length of the computation in FHE
test_data_length_full = 16000 #Number of clean samples possessed by the adversary

# Options: no real reason to change
show_mlir = False
batch_size = 32
test_batch_size = 32
use_cuda_if_available = True
seed = None
criterion = nn.CrossEntropyLoss()

# Seeding
if seed is None:
    seed = np.random.randint(0, 2**32 - 1)

print(f"\nUsing seed {seed}\n")
torch.manual_seed(seed);

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
train_loader, test_loader = manage_dataset(train_kwargs, test_kwargs)
img_size = train_loader.dataset.data[0].shape[0]

# Model definition
model = MNISTQATModel(a_bits, w_bits)
model = model.to(device)

# Training part
print(
    f"Performing MNIST task with {a_bits}-bits in activation quantization, {w_bits}-bits in weight quantization and a "
    f"sparsity of {sparsity}."
)

print("\n1. Loading pre-trained model")
# Ensure that "state_dict.pt" is pulled through GIT LFS
checkpoint = torch.load("state_dict.pt", map_location=device)
model.load_state_dict(checkpoint)

# Prepare compilation data
compilation_data = np.zeros((len(train_loader.dataset), img_size * img_size))
idx = 0

for data, target in tqdm(train_loader):
    for idx_batch, im in enumerate(data.numpy()):
        compilation_data[idx] = im
        idx += 1
        
# Test in the FHE simulation and in real FHE computation
accuracy = {}
current_index = 3

use_simulation = True

compile_and_test(
    model.cpu(),
    use_simulation,
    compilation_data,
    test_data_length_full,
    show_mlir,
    current_index,
)
    
        

