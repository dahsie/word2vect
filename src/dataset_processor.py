
import numpy as np
from typing import List
from nltk.tokenize import word_tokenize
import re


class Word2VectPreprocessor:
    """
    A class for text preprocessing tasks, including data cleaning, tokenization,
    vocabulary building, and creating training vectors for word embedding models.
    """

    def __init__(self):
        self.word2Ind = {}  # Map words to indices
        self.ind2Word = {}  # Map indices back to words
        self.V = None  # Vocabulary size (to be set later)

    def load_data(self, data_paths: List[str]):
        """
        Reads text data from a file, performs basic cleaning (lowercase, punctuation replacement),
        tokenization, and filtering (alphanumeric words and '.' only), and returns the cleaned list of words.

        Args:
        ----
            data_paths (List[str]): List of Path to the texts files.

        Returns:
        -------
            List[str]: List of cleaned and tokenized words.
        """
        for data_path in data_paths:
            data = []
            with open(data_path, 'r') as file:
                words = file.read()

            # Clean text: replace punctuations with '.', lowercase
            words = re.sub(r'[,!?;-]', '.', words)
            words = word_tokenize(words.lower())

            # Filter alphanumeric words and '.'
            data += [word for word in words if word.isalpha() or word == '.']
        return data

    def build_vocab(self, data):
        """
        Constructs a vocabulary from the processed text data (list of words).
        Creates dictionaries for mapping words to indices and vice versa.

        Args:
        ----
            data (List[str]): List of cleaned and tokenized words.
        """

        self.V = len(set(data))  # Vocabulary size
        self.word2Ind = {word: i for i, word in enumerate(sorted(set(data)))}
        self.ind2Word = {i: word for word, i in self.word2Ind.items()}

    def get_index(self, words: List[str]) -> List[int]:
        """
        Converts a list of words to their corresponding integer indices in the vocabulary.
        Returns a list of indices for words that exist in the vocabulary.

        Args:
        ----
            words (List[str]): List of words to convert to indices.

        Returns:
        -------
            List[int]: List of indices corresponding to the words.
        """

        return [self.word2Ind[word] for word in words if word in self.word2Ind]

    def index_frequencies(self, words):
        """
        Calculates the frequency of each word in a given list.
        Returns a dictionary where keys are words and values are their frequencies.

        Args:
        ----
            words (List[str]): List of words.

        Returns:
        -------
            Dict[str, int]: Dictionary of word frequencies.
        """

        frequencies = {}
        for word in words:
            frequencies[word] = frequencies.get(word, 0) + 1
        return frequencies

    def generate_vectors(self, data, context_window: int):
        """
        Generates training vectors for word embedding models using a sliding window approach.
        Yields pairs of (center word vector, context word vector) for each center word in the data.

        Args:
        ----
            data (List[str]): A list of cleaned and tokenized words.
            context_window (int): The size of the context window.

        Yields:
        ------
            Tuple[np.ndarray, np.ndarray]: A tuple containing the center word vector and the context vector.
        """
        if not self.word2Ind:
            raise ValueError("Vocabulary not built. Please call build_vocab first.")

        begin = context_window
        end = len(data) - context_window
        for i in range(begin, end):
        # while True:
            center = data[i]

            # Target vector (one-hot encoded)
            target = np.zeros(self.V)
            target[self.word2Ind[center]] = 1

            # Context vector (average frequency of surrounding words)
            context = data[(i - context_window):i] + data[i+1:(context_window + i + 1)]
            context_frequencies = self.index_frequencies(context)
            context_vector = np.zeros(self.V)
            for ind, freq in context_frequencies.items():
                if ind in self.word2Ind:
                    context_vector[self.word2Ind[ind]] = freq / len(context)
            yield context_vector, target

            #i += 1
            #if i >= len(data) - context_window:
            #    i = context_window

    def create_batch(self, data, context_window: int, batch_size: int):
        """
        Creates mini-batches of training vectors by iterating through the processed data with a sliding window.
        Yields batches of context vectors and target vectors (one-hot encoded).

        Args:
        ----
            data (List[str]): List of cleaned and tokenized words.
            context_window (int): The size of the context window.
            batch_size (int): The batch size.

        Yields:
        ------
            Tuple[np.ndarray, np.ndarray]: A tuple containing a batch of context vectors and a batch of target vectors.
        """

        batch_x = []
        batch_y = []
        for context_vector, target in self.generate_vectors(data, context_window):
            if len(batch_x) < batch_size:
                batch_x.append(context_vector)
                batch_y.append(target)
            else:
                yield np.array(batch_x).T, np.array(batch_y).T
                batch_x = []
                batch_y = []