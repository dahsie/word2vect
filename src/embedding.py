

import sys 
sys.path.append("/home/dah/nlp/word2vect/src/")

from word2vect import Word2vect
from typing import List, Dict
import numpy as np
from utils import *

class Embedding:
    
    def __init__(self,model: Word2vect, word2Ind:Dict[str, int],ind2Word: Dict[int, str], metric: str = "euclidean"):
        self.embeddings = (model.W1.T + model.W2)/2
        self.word2Ind = word2Ind
        self.ind2Word =ind2Word
        self.metric: str = metric

    def embedding(self,word: str) -> np.ndarray:
        idx = self.word2Ind[word.lower()]
        return self.embeddings[idx, :]
    
    def batch_embedding(self, words :List[str]):
        idx = [self.word2Ind[word.lower()] for word in words]
        return self.embeddings[idx, :]
    
    def embedding2word(self,embedding: np.ndarray) -> int:
        """
        Predicts the label for a single sample based on the nearest neighbors.

        Parameters:
        ----------
            x (np.ndarray): A single sample to predict the label for.

        Returns:
        -------
            int: Predicted label for the sample.
        """
        distances = None
        if self.metric == 'euclidean':
            distances = np.array([euclidean_distance(x1=embedding, x2=sample) for sample in self.embeddings])
        elif self.metric == 'cosine':
            distances = np.array([cosinus_similarity(x1=embedding, x2=sample) for sample in self.embeddings])
        nearest_index = list(np.argsort(a= distances)[:3])
        print(nearest_index)
        nearest_word = [self.ind2Word[index] for index in  nearest_index]
        return nearest_word
