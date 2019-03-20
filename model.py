from utils import add_layers


class Model:
    def __init__(self, sequence, activation_functions):
        self.depth = len(sequence) - 1
        self.sequence = sequence
        self.activation_functions = activation_functions

    def __call__(self, x):
        for i in range(self.depth):
            x = add_layers(x, self.sequence[i], self.sequence[i + 1], self.activation_functions[i])
        return x
