import numpy as np

class Sigmoid:

    def __call__(self, z):
        return 1/(1 + np.exp(-z))