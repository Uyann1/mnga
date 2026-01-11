
import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from mnga.autograd import Tensor
from mnga import nn
from mnga import optim
from mnga.optim import schedulers

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input 784 -> 128
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        # 128 -> 64
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        # 64 -> 10
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x # Return logits for CrossEntropyLoss

def accuracy(outputs, labels):
    # outputs: (B, 10) logits
    # labels: (B,) indices
    preds = np.argmax(outputs.data, axis=1)
    return np.mean(preds == labels)

def main():
    # 1. Load Data
    print("Loading MNIST data...")
    # Fetch data (this might take a while first time)
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X / 255.0 # Normalize
    y = y.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 2. Setup
    model = MNISTModel()
    
    # Optimizer (Adam)
    # Using default parameters lr=0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Scheduler: Decay LR by gamma every step_size epochs
    # For this demo, let's just do minor decay
    scheduler = schedulers.StepLR(optimizer, step_size=5, gamma=0.1)
    
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 64
    epochs = 5
    num_batches = X_train.shape[0] // batch_size
    
    print("Starting training...")
    
    for epoch in range(epochs):
        perm = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        start_time = time.time()
        
        for i in range(num_batches):
            idx = slice(i * batch_size, (i + 1) * batch_size)
            x_batch = Tensor(X_train_shuffled[idx])
            y_batch = y_train_shuffled[idx] # Keep as numpy array for Loss
            
            # Forward
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Step
            optimizer.step()
            
            # Record
            epoch_loss += loss.data
            epoch_acc += accuracy(outputs, y_batch)
            
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{num_batches}], Loss: {loss.data:.4f}, Acc: {accuracy(outputs, y_batch):.4f}")
        
        # Scheduler step at sample level or epoch level? PyTorch usually epoch.
        scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        duration = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{epochs}] Completed. Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}, Time: {duration:.2f}s, LR: {optimizer.lr}")
        
        # Validation
        val_loss = 0.0
        val_acc = 0.0
        val_steps = X_test.shape[0] // batch_size
        
        # Simple no_grad context? We don't have one yet, just don't call backward.
        # But we still build graph. For evaluation using Tensor without graph is cleaner but 
        # our layers output Tensors. We can just run it.
        
        for j in range(val_steps):
            idx = slice(j * batch_size, (j + 1) * batch_size)
            x_val = Tensor(X_test[idx])
            y_val = y_test[idx]
            
            out_val = model(x_val)
            l = criterion(out_val, y_val)
            val_loss += l.data
            val_acc += accuracy(out_val, y_val)
            
        print(f"Validation Loss: {val_loss/val_steps:.4f}, Acc: {val_acc/val_steps:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main()
