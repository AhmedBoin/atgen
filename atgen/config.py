import json
import pickle
import base64
from typing import Dict
from torch import nn
from layers import ActiSwitch, LayerModifier
from utils import evolve, follow, copy, skip

class ATGENConfig:
    def __init__(self, crossover_rate=0.8, crossover_decay=1.0, min_crossover=0.2, mutation_rate=0.03, mutation_decay=1.0, min_mutation=0.001, 
                 perturbation_rate=0.02, perturbation_decay=1.0, min_perturbation=0.02, wider_mutation=0.01, deeper_mutation=0.001, 
                 speciation_level=0, threshold=0.01, default_activation=ActiSwitch(nn.ReLU()), single_offspring=True, shared_fitness=True, 
                 dynamic_dropout_population=True, parent_mutation=True, remove_mutation=True, linear_start=True,
                 save_every_generation=True, extra_evolve=None, extra_follow=None, extra_copy=None, extra_skip=None):

        # Crossover setting
        self.crossover_rate = crossover_rate
        self.crossover_decay = crossover_decay
        self.min_crossover = min_crossover
        self.dynamic_dropout_population = dynamic_dropout_population
        self.single_offspring = single_offspring

        # Mutation setting
        self.mutation_rate = mutation_rate
        self.mutation_decay = mutation_decay
        self.min_mutation = min_mutation
        self.perturbation_rate = perturbation_rate
        self.perturbation_decay = perturbation_decay
        self.min_perturbation = min_perturbation
        # Extra Mutation setting
        self.wider_mutation = wider_mutation
        self.deeper_mutation = deeper_mutation
        self.remove_mutation = remove_mutation
        self.parent_mutation = parent_mutation

        self.speciation_level = speciation_level  # 0->layer level, else-> neuron level
        self.shared_fitness = shared_fitness
        self.save_every_generation = save_every_generation

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

    def perturbation_step(self):
        self.perturbation_rate = max(self.perturbation_decay * self.perturbation_rate, self.min_perturbation)

    def save(self, file_path: str):
        '''Saves the current configuration attributes and serialized layer dictionaries to a JSON file.'''
        actiswitch = isinstance(self.default_activation, ActiSwitch)
        if actiswitch:
            activation = self.default_activation.activation.__class__.__name__
        else:
            activation = self.default_activation.__class__.__name__ if self.default_activation is not None else None

        # Serialize the dictionaries using pickle
        modifiers = {
            'evolve': self.evolve,
            'follow': self.follow,
            'copy': self.copy,
            'skip': self.skip
        }

        serialized_data = pickle.dumps(modifiers)
        encoded_data = base64.b64encode(serialized_data).decode('utf-8')

        # Prepare JSON config data
        config_data = {
            'crossover_rate': self.crossover_rate,
            'crossover_decay': self.crossover_decay,
            'min_crossover': self.min_crossover,
            'dynamic_dropout_population': self.dynamic_dropout_population,
            'single_offspring': self.single_offspring,

            'mutation_rate': self.mutation_rate,
            'mutation_decay': self.mutation_decay,
            'min_mutation': self.min_mutation,
            'perturbation_rate': self.perturbation_rate,
            'perturbation_decay': self.perturbation_decay,
            'min_perturbation': self.min_perturbation,
            'wider_mutation': self.wider_mutation,
            'deeper_mutation': self.deeper_mutation,
            'remove_mutation': self.remove_mutation,
            'parent_mutation': self.parent_mutation,

            'speciation_level': self.speciation_level,
            'shared_fitness': self.shared_fitness,
            'save_every_generation': self.save_every_generation,

            'threshold': self.threshold,
            'actiswitch': actiswitch,
            'activation': activation,
            'linear_start': self.linear_start,
            'modifiers': encoded_data,  # This is binary data encoded as a base64 string. Do not modify.
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
        encoded_data = config_data['modifiers']
        serialized_data = base64.b64decode(encoded_data)
        modifiers = pickle.loads(serialized_data)

        return cls(
            crossover_rate=config_data['crossover_rate'],
            crossover_decay=config_data['crossover_decay'],
            min_crossover=config_data['min_crossover'],
            dynamic_dropout_population=config_data['dynamic_dropout_population'],
            single_offspring=config_data['single_offspring'],

            mutation_rate=config_data['mutation_rate'],
            mutation_decay=config_data['mutation_decay'],
            min_mutation=config_data['min_mutation'],
            perturbation_rate=config_data['perturbation_rate'],
            perturbation_decay=config_data['perturbation_decay'],
            min_perturbation=config_data['min_perturbation'],
            wider_mutation=config_data['wider_mutation'],
            deeper_mutation=config_data['deeper_mutation'],
            remove_mutation=config_data['remove_mutation'],
            parent_mutation=config_data['parent_mutation'],

            speciation_level=config_data['speciation_level'],
            shared_fitness=config_data['shared_fitness'],
            save_every_generation=config_data['save_every_generation'],

            threshold=config_data['threshold'],
            default_activation=default_activation,
            linear_start=config_data['linear_start'],
            extra_evolve=modifiers['evolve'],
            extra_follow=modifiers['follow'],
            extra_copy=modifiers['copy'],
            extra_skip=modifiers['skip']
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