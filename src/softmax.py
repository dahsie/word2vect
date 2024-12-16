import numpy as np

class Softmax:

    def __call__(self,z):
        total = np.exp(z)
        yhat = total/total.sum(axis=0)
        return yhat
