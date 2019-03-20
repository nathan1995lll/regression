import numpy as np


class DataGenerator:
    def __init__(self, n_samples, white_noise, function):
        self.n_samples = n_samples
        self.white_noise = white_noise
        self.function = function

    def __call__(self):
        x = np.linspace(-10, 10, self.n_samples).reshape([-1, 1])
        noise = np.random.normal(loc=self.white_noise[0], scale=self.white_noise[1], size=self.n_samples).reshape(
            [-1, 1])
        y = self.function(x) + noise
        return x, y

    def __str__(self):
        return "function : {}, noise : {}".format(self.function, self.white_noise)
