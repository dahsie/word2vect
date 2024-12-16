


class Trainer:
    
    def __init__(self, model, optimizer, loss_fn, data_processor):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.data_processor = data_processor

    def train(self, data, epochs, batch_size, context_window):
        
        for epoch in range(epochs):
            for x, y in self.data_processor.create_batch(data = data, context_window = context_window, batch_size = batch_size):
                h, yhat = self.model(x)  # Forward pass
                loss = self.loss_fn(yhat=yhat,y=y, batch_size = batch_size)
                
                # Calculate gradients
                gradients = self.model.backward(X_batch = x, batch_size = batch_size, Y = y, Yhat = yhat)

                # Update weights
                self.optimizer.step(model = self.model, gradients=gradients)

            # Print loss (optional)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss:.6f}")
            if (epoch + 1) % 50 == 0:
                self.optimizer.learning_rate *= 0.5
                print(f"learning rate : {self.optimizer.learning_rate}")
        
  
            
