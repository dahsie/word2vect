import numpy as np


class CrossEntropyLoss:

    def __call__(self, y, yhat, batch_size):
        return -(1/batch_size) * np.sum(y * np.log(yhat))