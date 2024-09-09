# import copy
# import random

# import torch

# from layers import ActiSwitch, Conv2D, Linear, Flatten


class DNA:
    def __init__(self, input_size=None):
        self.input_size = input_size
        self.flatten = None
        self.conv = []
        self.maxpool = []
        self.linear = []
    
    def append_conv(self, conv):
        self.conv.append(conv)
    
    def append_linear(self, linear):
        self.linear.append(linear)
    
    def append_maxpool(self, maxpool):
        self.maxpool.append(maxpool)
    
    # def __add__(self, dna: "DNA"):
    #     new_dna = DNA(self.input_size)
    #     new_dna.flatten = self.flatten
    #     if self.conv:
    #         new_dna.conv, new_dna.maxpool = conv_crossover(self.conv, dna.conv, self.maxpool)
    #     if self.linear:
    #         new_dna.linear = linear_crossover(self.linear, dna.linear)
    #     return new_dna
    
    def __str__(self) -> str:
        return f"""DNA Sequence:
        Input Size: {self.input_size}
        {f"Conv: {len(self.conv)}" if self.conv else ""}
        {f"MaxPool: {len(self.maxpool)}" if self.maxpool else ""}
        {f"Linear: {len(self.linear)}" if self.linear else ""}"""
    
    def reconstruct(self):
        layers = []
        if self.conv:
            for layer in self.conv:
                layers.extend(layer)
        if self.maxpool:
            for layer in self.maxpool:
                layers.insert(*layer)
        if self.flatten is not None:
            layers.append(self.flatten)
        if self.linear:
            for layer in self.linear:
                layers.extend(layer)
        print(layers)
        return layers
    
    def linear_size(self):
        return [self.linear[0][0].in_features, *[layer[0].out_features for layer in self.linear]]
    
    def conv_size(self):
        return [self.conv[0][0].in_channels, *[layer[0].out_channels for layer in self.conv]]
    
    

# # Helper functions
# def get_removal_indices_for_larger_list(original_list, required_items):
#     return sorted(random.sample(range(len(original_list)), len(original_list)-required_items), reverse=True)

# def get_inserted_indices_for_shorter_list(original_list, required_items):
#     return sorted(random.sample(range(1, required_items-1), required_items-len(original_list)), reverse=True)
    
# def network_arranger(net1, net2):
#     l1, l2 = len(net1), len(net2)
#     if l1 > l2:
#         longer_net, shorter_net = net1, net2
#     else:
#         shorter_net, longer_net = net1, net2
#     l = random.randint(min(l1, l2), max(l1, l2))
#     longer_indices = get_removal_indices_for_larger_list(longer_net, l)
#     shorter_indices = get_inserted_indices_for_shorter_list(shorter_net, l)
#     return longer_net, shorter_net, longer_indices, shorter_indices

# # Network crossover
# def conv_crossover(conv1, conv2, maxpool):
#     new_conv = []
#     new_maxpool = []
#     longer_net, shorter_net, longer_indices, shorter_indices = network_arranger(conv1, conv2)
#     shorter_net: list
#     longer_net: list

#     '''handle lengths and weights'''
#     for idx in longer_indices:
#         longer_net.pop(idx)

#     for idx in shorter_indices:
#         layer, activation = shorter_net[idx-1][0], shorter_net[idx-1][1]
#         shorter_net.insert(idx, (
#             Conv2D.init_identity_layer(layer.out_channels, layer.kernel_size, True if layer.bias is not None else False, layer.norm), 
#             ActiSwitch(activation.activation, True)
#             ))
    
#     for i in range(len(longer_net)):
#         new_conv.append(conv_crossover_layer(longer_net[i], shorter_net[i]))

#     maxpool_indices = get_inserted_indices_for_shorter_list(longer_net, len(maxpool))
#     for idx in maxpool_indices:
#         new_maxpool.append((idx, maxpool.pop()[1]))
    
#     return new_conv, new_maxpool

# def linear_crossover(linear1, linear2):
#     new_linear = []
#     longer_net, shorter_net, longer_indices, shorter_indices = network_arranger(linear1, linear2)

#     '''handle lengths and weights'''
#     for idx in longer_indices:
#         longer_net.pop(idx)

#     for idx in shorter_indices:
#         layer, activation = shorter_net[idx-1][0], shorter_net[idx-1][1]
#         shorter_net.insert(idx, (
#             Linear.init_identity_layer(layer.out_features, True if layer.bias is not None else False, layer.norm_type), 
#             ActiSwitch(activation.activation, True)
#             ))
    
#     for i in range(len(longer_net)):
#         new_linear.append(linear_crossover_layer(longer_net[i], shorter_net[i]))
#     return new_linear

# # Layers crossover
# def conv_crossover_layer(conv1: Conv2D, conv2: Conv2D):
#     if conv1.in_channels > conv2.in_channels:
#         for _ in range(conv1.in_channels-conv2.in_channels):
#             conv1.add_input_channel()
#     if conv2.in_channels > conv1.in_channels:
#         for _ in range(conv1.in_channels-conv2.in_channels):
#             conv2.add_input_channel()
#     if conv1.out_channels > conv2.out_channels:
#         for _ in range(conv1.out_channels-conv2.out_channels):
#             conv1.add_output_channel()
#     if conv2.out_channels > conv1.out_channels:
#         for _ in range(conv1.out_channels-conv2.out_channels):
#             conv2.add_output_channel()
            
#     # Apply random mask to weights and biases for crossover
#     mask = torch.rand_like(conv1.weight.data, device=conv1.weight.device) < 0.5
#     offspring_layer_weight = torch.where(mask, conv1.weight.data, conv2.weight.data)

# def linear_crossover_layer(linear1: Linear, linear2: Linear):
#     pass