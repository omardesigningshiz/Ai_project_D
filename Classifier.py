# Hand Sign Classifier Training Script
# This script trains a neural network to classify Arabic hand signs based on the preprocessed landmark data.
# It uses GPU acceleration if available and includes a train/validation split to monitor performance during training.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle

# 1. Setup Device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# 2. Load the preprocessed data
with open('data.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Convert lists to PyTorch tensors
data = torch.tensor(dataset['data'], dtype=torch.float32)
labels = torch.tensor(dataset['labels'], dtype=torch.long)

# Create the full TensorDataset
full_dataset = TensorDataset(data, labels)

# 3. The Train/Validation Split (80% Train, 20% Validation)
dataset_size = len(full_dataset)
val_size = int(0.2 * dataset_size)
train_size = dataset_size - val_size

# Randomly split the dataset into training and validation sets
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 4. Define the Neural Network Model


class HandSignClassifier(nn.Module):
    def __init__(self):
        super(HandSignClassifier, self).__init__()
        # 42 input features (21 landmarks * 2 for x and y) and 30 output classes (Arabic letters)
        # 3 layers with ReLU activations and dropout for regularization
        self.fc1 = nn.Linear(42, 128)
        # Dropout layer with 20% dropout rate to prevent overfitting
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 30)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        # No activation here because CrossEntropyLoss expects raw logits
        x = self.fc3(x)
        return x


if __name__ == "__main__":   # Ensure this block only runs when the script is executed directly, not when imported as a module
    # Initialize the model and move it to the GPU
    model = HandSignClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Training and Validation Loop
    num_epochs = 20

    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0
        correct_train = 0

        for batch_data, batch_labels in train_loader:
            # Move data to GPU
            batch_data, batch_labels = batch_data.to(
                device), batch_labels.to(device)

            optimizer.zero_grad()  # Clear gradients from the previous step
            # Forward pass through the model to get predictions
            outputs = model(batch_data)
            # Calculate the loss between predictions and true labels
            loss = criterion(outputs, batch_labels)
            loss.backward()  # Backpropagation to compute gradients
            optimizer.step()  # Update model parameters based on computed gradients

            total_train_loss += loss.item()  # Accumulate the training loss for this batch

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == batch_labels).sum().item()

        # Average training loss for this epoch
        avg_train_loss = total_train_loss / len(train_loader)
        # Calculate training accuracy as a percentage
        train_accuracy = 100 * correct_train / train_size

        # --- VALIDATION PHASE ---
        model.eval()  # Turn off dropout for validation
        total_val_loss = 0
        correct_val = 0

        with torch.no_grad():
            # Loop through validation batches without tracking gradients and move data to GPU
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(
                    device), batch_labels.to(device)

                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                total_val_loss += loss.item()

                # Get the predicted class with the highest score
                _, predicted = torch.max(outputs.data, 1)
                # Count how many predictions were correct
                correct_val += (predicted == batch_labels).sum().item()

        # Average validation loss for this epoch
        avg_val_loss = total_val_loss / len(val_loader)
        # Calculate validation accuracy as a percentage
        val_accuracy = 100 * correct_val / val_size

        # Print progress
        print(f'Epoch [{epoch+1}/{num_epochs}] | '
              f'Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}% | '
              f'Val Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.2f}%')

    # Save the trained model
    torch.save(model.state_dict(), 'hand_sign_classifier.pth')
    print("\nModel saved successfully!")
