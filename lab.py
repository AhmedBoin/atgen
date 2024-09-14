import math
import torch
from torch import nn

# print(nn.Linear(10, 1).weight.shape[1])

# model = nn.Sequential(
#     nn.Linear(5, 10),
#     nn.ReLU(),
#     nn.Linear(10, 5),
#     nn.ReLU(),
#     nn.Linear(5, 1)
# )

# for layer in model:
#     print(layer)

# print()
# print()
# model.insert(0, nn.ReLU())

# for layer in model:
#     print(layer)

# model.pop(0)
# model.pop(1)
# model.pop(2)
# print()
# print()
# for layer in model:
#     print(layer)

# model.apply()

model = nn.Sequential(
    nn.Linear(5, 3),
    nn.ReLU(),
    nn.Linear(3, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
)

evolve_layers = [nn.Linear]
for layer in model:
    for layer_type in evolve_layers:
        if isinstance(layer, layer_type):
            print(layer)



def splitter(model: nn.Sequential):
    modules = []
    dna = []
    for layer in model:
        for layer_type in evolve_layers:
            if isinstance(layer, layer_type):
                if modules:
                    dna.append(modules)
                modules = []
        modules.append(layer)
    if modules:
        dna.append(modules)
    return dna

dna = splitter(model)
print(dna)

layer = nn.LazyLinear(2)
layer(torch.rand(64, 10))
print(layer.in_features)
print(type(layer))
print(layer.__class__ == nn.Linear)

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
    nn.ReLU(True),
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), # 14x14 -> 7x7
    nn.ReLU(True),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # 7x7 -> 4x4
    nn.ReLU(True),
    nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1), # 4x4 -> 7x7
    nn.ReLU(True),
    nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1), # 7x7 -> 14x14
    nn.ReLU(True),
    nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
    nn.Sigmoid()  # Output in range [0, 1] for normalized data (like images)
)

layer = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
print(layer.weight.shape, layer.bias.shape)

print(hash("Conv2dConv2dConv2dConvTranspose2dConvTranspose2dConvTranspose2dConvTranspose2dConvTranspose2d"))
print(hash("loll"))
print(hash("Hi"))

x = [1, 2, 3]
print(x[:-1])


from tqdm import tqdm

# Example dictionary
my_dict = {
    'a': [1, 2, 3],
    'b': [4, 5, 6],
    'c': [7, 8, 9]
}

# Calculate the total number of items in all lists
total_items = sum(len(value_list) for value_list in my_dict.values())

# Create a single progress bar for all items
with tqdm(total=total_items) as pbar:
    for value_list in my_dict.values():
        for i in range(len(value_list)):
            value_list[i] = 5  # Modify the value in place
            pbar.update(1)  # Update the progress bar

print(my_dict)
print(math.log2(1.0+1))

print(nn.Linear(10, 1).parameters().__next__().requires_grad)

x = nn.Module().parameters()
print(x)