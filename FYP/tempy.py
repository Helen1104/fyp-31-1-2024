
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

root = os.path.dirname(os.path.abspath(__file__))

def file_combined(root):

    directory = os.path.join(root,'data')

    excel_files = [file for file in os.listdir(directory) if file.endswith(".xlsx")]

    dfs = []

    for file in excel_files:
        file_path = os.path.join(directory, file)
        
        df = pd.read_excel(file_path)
        df.iloc[0] = df.iloc[0].astype(str)
        #df = df.reset_index(drop=True)
        df = df.rename(columns=df.iloc[0]).drop(df.index[0])
        
        dfs.append(df.reset_index(drop=True))

    # Concatenate all dataframes into a single dataframe
    combined_df = pd.concat(dfs, ignore_index=True)
    #combined_df = combined_df.reset_index(drop=True)

     # Drop duplicate rows based on all columns
    combined_df = combined_df.drop_duplicates()

    # Reset the index and assign a new unique index
    combined_df = combined_df.reset_index(drop=True)
    #combined_df.index = pd.RangeIndex(start=0, stop=len(combined_df))

    print(combined_df.index)  # Print the index values

    # Save the combined dataframe to a CSV file
    combined_df.to_csv(os.path.join(root, 'data_modified', 'combined.csv'), index=False)

    return combined_df

def main():
    combined_df = file_combined(root)
    print(combined_df)

    # Step 1: Load the Excel file using pandas
    df = combined_df

    # Step 2: Preprocess the data, if needed
    # Perform any necessary data cleaning, feature engineering, or normalization

    # Step 3: Define a custom dataset class
    class CustomDataset(Dataset):
        def __init__(self, data):
            self.xy = data.iloc[:, :-1].values.values.astype(np.float32)
            self.labels = data.iloc[:, -1].values
            
        def __len__(self):
            return len(self.xy)
        
        def __getitem__(self, index):
            x = torch.Tensor(self.xy[index].astype(np.float32))
            y = torch.Tensor([self.labels[index]])
            return x, y

    # Step 4: Create train and test datasets
    dataset = CustomDataset(df)
    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    # Step 5: Define a simple neural network model
    class NeuralNetwork(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Step 6: Create an instance of the neural network model
    input_size = len(df.columns) - 1
    hidden_size = 64
    output_size = 1
    model = NeuralNetwork(input_size, hidden_size, output_size)

    # Step 7: Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Step 8: Create data loaders for training and testing
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Step 9: Train the neural network
    num_epochs = 10
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # Step 10: Evaluate the neural network
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
    average_loss = total_loss / len(test_loader)
    print(f"Average Loss: {average_loss}")


if __name__ == "__main__":
    main()