import json
import pickle
import base64
from typing import Dict
from torch import nn
from layers import ActiSwitch, LayerModifier
from utils import evolve, follow, copy, skip

class ATGENConfig:
    def __init__(self, crossover_rate=0.8, crossover_decay=1.0, min_crossover=0.2, 
                 mutation_rate=0.1, perturbation_rate=0.9, mutation_decay=1.0, min_mutation=0.001, wider_mutation=0.01, 
                 deeper_mutation=0.0001, threshold=0.01, default_activation=ActiSwitch(nn.ReLU()), 
                 dynamic_dropout_population=True, parents_mutate=True, remove_mutation=True, linear_start=True, 
                 extra_evolve=None, extra_follow=None, extra_copy=None, extra_skip=None):

        # Crossover setting
        self.crossover_rate = crossover_rate
        self.crossover_decay = crossover_decay
        self.min_crossover = min_crossover
        self.dynamic_dropout_population = dynamic_dropout_population

        # Mutation setting
        self.mutation_rate = mutation_rate
        self.perturbation_rate = perturbation_rate
        self.mutation_decay = mutation_decay
        self.min_mutation = min_mutation
        # Extra Mutation setting
        self.parents_mutate = parents_mutate
        self.wider_mutation = wider_mutation
        self.deeper_mutation = deeper_mutation
        self.remove_mutation = remove_mutation

        # Network setting
        self.threshold = threshold
        self.default_activation = default_activation
        self.linear_start = linear_start
        self.evolve: Dict[nn.Module, LayerModifier] = evolve
        self.follow: Dict[nn.Module, LayerModifier] = follow
        self.copy: Dict[nn.Module, LayerModifier] = copy
        self.skip: Dict[nn.Module, LayerModifier] = skip

        # Evolving setting
        if extra_evolve is not None:
            self.evolve.update(extra_evolve)
        if extra_follow is not None:
            self.follow.update(extra_follow)
        if extra_copy is not None:
            self.copy.update(extra_copy)
        if extra_skip is not None:
            self.skip.update(extra_skip)

    def crossover_step(self):
        self.crossover_rate = max(self.crossover_decay * self.crossover_rate, self.min_crossover)

    def mutation_step(self):
        self.mutation_rate = max(self.mutation_decay * self.mutation_rate, self.min_mutation)

    def save(self, file_path: str):
        '''Saves the current configuration attributes and serialized layer dictionaries to a JSON file.'''
        actiswitch = isinstance(self.default_activation, ActiSwitch)
        if actiswitch:
            activation = self.default_activation.activation.__class__.__name__
        else:
            activation = self.default_activation.__class__.__name__ if self.default_activation is not None else None

        # Serialize the dictionaries using pickle
        layer_dicts = {
            'evolve': self.evolve,
            'follow': self.follow,
            'copy': self.copy,
            'skip': self.skip
        }

        serialized_data = pickle.dumps(layer_dicts)
        encoded_data = base64.b64encode(serialized_data).decode('utf-8')

        # Prepare JSON config data
        config_data = {
            'crossover_rate': self.crossover_rate,
            'crossover_decay': self.crossover_decay,
            'min_crossover': self.min_crossover,
            'dynamic_dropout_population': self.dynamic_dropout_population,
            'mutation_rate': self.mutation_rate,
            'perturbation_rate': self.perturbation_rate,
            'mutation_decay': self.mutation_decay,
            'min_mutation': self.min_mutation,
            'wider_mutation': self.wider_mutation,
            'deeper_mutation': self.deeper_mutation,
            'parents_mutate': self.parents_mutate,
            'remove_mutation': self.remove_mutation,
            'threshold': self.threshold,
            'actiswitch': actiswitch,
            'activation': activation,
            'linear_start': self.linear_start,
            'layer_dicts': encoded_data,  # This is binary data encoded as a base64 string. Do not modify.
        }

        with open(file_path, 'w') as f:
            json.dump(config_data, f, indent=4)

    @classmethod
    def load(cls, file_path: str) -> 'ATGENConfig':
        '''Loads a configuration from a JSON file and deserializes layer dictionaries from the encoded data.'''
        with open(file_path, 'r') as f:
            config_data = json.load(f)

        # Handle activation loading
        default_activation = cls._load_activation(config_data['actiswitch'], config_data['activation'])

        # Decode and deserialize the layer dictionaries
        encoded_data = config_data['layer_dicts']
        serialized_data = base64.b64decode(encoded_data)
        layer_dicts = pickle.loads(serialized_data)

        return cls(
            crossover_rate=config_data['crossover_rate'],
            crossover_decay=config_data['crossover_decay'],
            min_crossover=config_data['min_crossover'],
            dynamic_dropout_population=config_data['dynamic_dropout_population'],
            mutation_rate=config_data['mutation_rate'],
            perturbation_rate=config_data['perturbation_rate'],
            mutation_decay=config_data['mutation_decay'],
            min_mutation=config_data['min_mutation'],
            wider_mutation=config_data['wider_mutation'],
            deeper_mutation=config_data['deeper_mutation'],
            parents_mutate=config_data['parents_mutate'],
            remove_mutation=config_data['remove_mutation'],
            threshold=config_data['threshold'],
            default_activation=default_activation,
            linear_start=config_data['linear_start'],
            extra_evolve=layer_dicts['evolve'],
            extra_follow=layer_dicts['follow'],
            extra_copy=layer_dicts['copy'],
            extra_skip=layer_dicts['skip']
        )

    @classmethod
    def _load_activation(cls, actiswitch, activation_name):
        '''Helper method to load the activation function from the config.'''
        if actiswitch:
            activation_class = getattr(nn, activation_name)
            return ActiSwitch(activation_class())
        elif activation_name:
            activation_class = getattr(nn, activation_name)
            return activation_class()
        return None


if __name__ == "__main__":
    config = ATGENConfig()
    config.save('config.json')
    loaded_config = ATGENConfig.load('config.json')
    print(loaded_config.__dict__)