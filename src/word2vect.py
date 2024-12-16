
import numpy as np

import sys
sys.path.append("/home/dah/nlp/word2vect/src/")

from softmax import Softmax
from relu import ReLU

class Word2vect:

    def __init__(self, embedding_dim:int, vocab_size:int, random_state: int = 42):

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
    
        self.random_state = random_state

        self.W1 = None
        self.b1 = None
        self.relu = ReLU()
        self.W2 = None
        self.b2 = None
        self.softmax = Softmax()

        self.initialization()

    def initialization(self):

        np.random.seed(self.random_state)
        self.W1 = np.random.rand(self.embedding_dim, self.vocab_size)
        self.b1 = np.random.rand(self.embedding_dim, 1)
        
        self.W2 = np.random.rand(self.vocab_size, self.embedding_dim)
        self.b2 = np.random.rand(self.vocab_size, 1)

    
    
    def forward(self, x):
        h = self.W1.dot(x) + self.b1
        h = self.relu(h) #  activation1

        z = self.W2.dot(h) + self.b2
        z = self.softmax(z) # activation 2
        
        return h, z

    def backward(self, X_batch: np.ndarray, batch_size: int, Y:np.ndarray, Yhat:np.ndarray):

        diff = Yhat - Y
        Z = self.W1.dot(X_batch) #
        H = self.relu(Z)
        
        dw1 =  (self.W2.T.dot(diff) * self.relu.derivative(Z)).dot(X_batch.T)/batch_size
        db1 =  np.sum(a = self.W2.T.dot(diff) * self.relu.derivative(Z), axis=1, keepdims=True)/batch_size 
        dw2 = diff.dot(H.T)/batch_size
        db2 = np.sum(diff, axis=1, keepdims=True)/batch_size

        return {"W1": dw1,"b1": db1,"W2": dw2,"b2": db2}
    
    def __call__(self, x):
        return self.forward(x)
        
        