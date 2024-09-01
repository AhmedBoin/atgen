import pickle
import torch
from torch import nn

from typing import List
import random
from utils import BLUE, BOLD, RESET_COLOR
from layers import Linear, ActiSwitch, Pass

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class ATNetwork(nn.Module):
    """
    ATNetwork: A neural network model capable of evolving its structure and weights.
    This class provides methods to dynamically change the network architecture, 
    such as adding or removing layers, evolving weights, and modifying activation functions.
    """
    def __init__(self, layers: List[int]=None, activation=nn.ReLU(), last_activation=None, bias=True, backprob_phase=True):
        """
        Initialize the ATNetwork.

        Args:
            layers (List[int], optional): A list defining the number of neurons in each layer.
            activation (nn.Module, optional): The default activation function to use in all layers except the last. Defaults to nn.ReLU().
            last_activation (nn.Module, optional): The activation function to use in the last layer. Defaults to None (no activation).
            bias (bool, optional): Whether to include bias terms in the layers. Defaults to True.
            backprob_phase (bool, optional): Indicates if back propagation learning-based evolution is enabled. Defaults to True.

        Raises:
            NotImplementedError: If the network is not properly initialized through the constructor.
        """
        super(ATNetwork, self).__init__()
        if layers:
            self.layers = nn.ModuleList([Linear(layers[idx], layers[idx + 1], bias) for idx in range(len(layers) - 1)])
            self.activation = nn.ModuleList([
                *[ActiSwitch(activation, backprob_phase) for _ in range(len(layers) - 2)], 
                Pass() if last_activation is None else last_activation
            ])
            self.backprob_phase = backprob_phase
            self.default_activation = activation
        else:
            raise NotImplementedError("you should implement the Network using constructor or inherit class and make your own implementation")


    def forward(self, x: torch.Tensor):
        for i in range(len(self.layers)):
            x = self.activation[i](self.layers[i](x))
        return x

    def evolve_network(self, idx=None):
        """
        Evolve the network by adding a new layer with identity initialization.

        Args:
            idx (int, optional): The index at which to insert the new layer. If None, a random index is chosen.

        Modifies:
            The network architecture is modified in place with an additional layer and activation function.
        """
        idx = random.randint(0, len(self.layers)-1) if idx is None else idx
        self.layers.insert(idx, Linear.init_identity_layer(self.layers[idx].in_features, True if self.layers[idx].bias is not None else False))
        self.activation.insert(idx, ActiSwitch(self.default_activation, self.backprob_phase))

    def evolve_layer(self, idx=None):
        """
        Add a new neuron to an existing layer and adjust subsequent layers accordingly.

        Args:
            idx (int, optional): The index of the layer to evolve. If None, a random layer is selected.

        Modifies:
            The specified layer is expanded by one neuron, and the next layer's input dimension is increased by one.
        """
        if len(self.layers) > 1: # to avoid changing output layer shape
            idx = random.randint(0, len(self.layers) - 2) if idx is None else idx
            self.layers[idx].add_neuron()
            if idx < len(self.layers) - 1: # avoid out of range, this case handle crossover during the process of adding neuron to output layer
                self.layers[idx + 1].add_weight()
        elif idx is not None: # handle if only 1 output layer in crossover
            self.layers[idx].add_neuron()
        # self.summary()

    @torch.no_grad()
    def evolve_weight(self, mutation_rate, perturbation_rate):
        """
        Apply random noise to the network weights to facilitate evolutionary adaptation.

        Args:
            mutation_rate (float): The probability of applying noise to each parameter.
            perturbation_rate (float): The magnitude of noise to apply.

        Modifies:
            Adds Gaussian noise to the weights of the network with a probability defined by 'mutation_rate'.
        """
        for param in self.layers.parameters():
            if random.random() < mutation_rate:
                noise = torch.randn_like(param) * perturbation_rate  # Adjust perturbation magnitude
                param.add_(noise)
        # self.summary()


    def evolve_activation(self, activation_dict: List[nn.Module], idx=None):
        """
        Change the activation function of a specified layer to a new random function from a given list.

        Args:
            activation_dict (List[nn.Module]): A list of possible activation functions to choose from.
            idx (int, optional): The index of the layer whose activation function should be changed. 
                If None, a random layer is selected.

        Modifies:
            Replaces the activation function of the specified layer with a randomly selected function from 'activation_dict'.
        """
        if len(self.activation) > 1: # to avoid changing output layer activation
            idx = random.randint(0, len(self.layers)-2) if idx is None else idx
            activation = random.choice(activation_dict)
            self.activation[idx].change_activation(activation)
            # self.summary()

    def prune(self, threshold: float = 0.01):
        """
        Prune neurons with weights below a given threshold.
        
        Args:
            threshold (float): The threshold below which neurons will be pruned.
        """
        for i, layer in enumerate(self.layers[:-1]):
            if (isinstance(layer, Linear) or str(type(layer)) == "<class 'layers.Linear'>") and layer.out_features > 1:
                try:
                    # Identify neurons to prune
                    neurons_to_prune = []
                    for neuron_idx in range(layer.out_features):
                        neuron_weights = layer.weight[neuron_idx].abs()
                        if torch.max(neuron_weights) < threshold:
                            neurons_to_prune.append(neuron_idx)
                    
                    for neuron_idx in reversed(neurons_to_prune):
                        layer.remove_neuron(neuron_idx)
                        self.layers[i + 1].remove_weight(neuron_idx)
                except:
                    pass

    def genome_type(self) -> List[int]:
        return [layer.out_features for layer in self.layers]
    
    def save_network(self, file_name="ATNetwork.pth"):
        with open(f'{file_name}', 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_network(cls, file_name="ATNetwork.pth") -> "ATNetwork":
        with open(f'{file_name}', 'rb') as file:
            network: ATNetwork = pickle.load(file)
        return network

    @torch.no_grad()
    def summary(self):
        print(f"{BOLD}{BLUE}Model Summary{RESET_COLOR}{BOLD}:")  # Bold text for the title
        print("-" * 85)

        # Define and print column headers
        headers = f"{'Layer':<11}{'Output Shape':<30}{'Parameters':<15}{'Activation':<15}"
        print(headers)
        print("-" * 85)

        total_param = 0
        for i, (layer, activation) in enumerate(zip(self.layers, self.activation)):
            total_param += layer.params_count
            
            # Format each layer's information with fixed-width columns
            try:
                layer_info = f"Layer {i+1:<5}{(f'(batch_size, {layer.out_features})'):<30}{layer.params_count:<15}{activation.__class__.__name__}({activation.activation.__class__.__name__}, {100*(activation.activation_weight.item()/(activation.linear_weight.item()+activation.activation_weight.item())):.2f}%)"
            except:
                layer_info = f"Layer {i+1:<5}{(f'(batch_size, {layer.out_features})'):<30}{layer.params_count:<15}{activation.__class__.__name__:<15}"
            print(layer_info)

        print("-" * 85)
        print(f"{BLUE}{'Total Parameters:':<25}{RESET_COLOR}{BOLD}{total_param:<15}{RESET_COLOR}")




if __name__ == "__main__":
    # helper loss function
    def loss(x1:torch.Tensor, x2: torch.Tensor):
        val = torch.abs((x2/x1).mean() - 1) * 100
        print(f"loss = {val:.10f}%")


    model = ATNetwork([5, 3, 2, 1])
    class CustomNetwork(ATNetwork):
        def __init__(self):
            # Do not call the parent __init__ with layers; initialize manually, Directly call nn.Module's init
            super(ATNetwork, self).__init__()
            self.layers = nn.ModuleList([
                Linear(5, 3),
                Linear(3, 1)
            ])
            self.activation = nn.ModuleList([
                ActiSwitch(),
                Pass()
            ])
            self.backprob_phase = True
            self.default_activation = nn.ReLU()

    model = CustomNetwork()
    model.summary()
    # summary(model, input_size=(5,))

    x = torch.randn(4, 5)
    y1: torch.Tensor = model(x)
    
    for _ in range(10):
        model.evolve_network()
    model.summary()
    # summary(model, input_size=(5,))
    for _ in range(500):
        model.evolve_layer()
    model.summary()
    # summary(model, input_size=(5,))
        
    y2: torch.Tensor = model(x)
    model.prune()
    model.summary()
    # summary(model, input_size=(5,))

    print(torch.cat((y1, y2),  dim=1))  
    loss(y1, y2)





