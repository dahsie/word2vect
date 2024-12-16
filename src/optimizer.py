

class Optimizer:
    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate

    def step(self, model, gradients):
        """
        Met à jour les poids du modèle en utilisant les gradients calculés.

        Args:
            gradients: Un dictionnaire contenant les gradients pour chaque paramètre.
        """
        for param_name, grad in gradients.items():
            setattr(model, param_name, getattr(model, param_name) - self.learning_rate * grad)