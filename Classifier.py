# Hand Sign Classifier Training Script
# This script trains a neural network to classify Arabic hand signs based on the preprocessed landmark data.
# It uses GPU acceleration if available and includes a train/validation split to monitor performance during training.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

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


if __name__ == "__main__":
    # Initialize the model and move it to the GPU
    model = HandSignClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Training and Validation Loop
    num_epochs = 20

    #  Lists to track metrics for plotting
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0
        correct_train = 0

        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(
                device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == batch_labels).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / train_size

        # --- VALIDATION PHASE ---
        model.eval()
        total_val_loss = 0
        correct_val = 0

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(
                    device), batch_labels.to(device)

                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == batch_labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / val_size

        #  Store metrics for this epoch
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}] | '
              f'Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}% | '
              f'Val Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.2f}%')

    # Save the trained model
    torch.save(model.state_dict(), 'hand_sign_classifier.pth')
    print("\nModel saved successfully!")

    # 5. VISUALIZATION & PERFORMANCE EVALUATION

    print("\nGenerating performance metrics and plots...")

    # A. Convergence Plots (Loss and Accuracy)
    plt.figure(figsize=(14, 5))

    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy', color='blue')
    plt.plot(val_accs, label='Validation Accuracy', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('convergence_plots.png', dpi=300)
    print("-> Saved 'convergence_plots.png'")

    # B. Advanced Metrics & Confusion Matrix
    model.eval()
    all_preds = []
    all_labels = []

    # Run the entire validation set through the model one last time
    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            batch_data = batch_data.to(device)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    # Print the Classification Report (Precision, Recall, F1 for every single letter)
    print("\n--- Detailed Classification Report ---")
    print(classification_report(all_labels, all_preds, zero_division=0))

    # Generate and save the Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title('Confusion Matrix: Arabic Hand Signs')
    plt.xlabel('Predicted Class (0-29)')
    plt.ylabel('True Class (0-29)')
    plt.savefig('confusion_matrix.png', dpi=300)
    print("-> Saved 'confusion_matrix.png'")
