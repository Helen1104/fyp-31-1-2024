
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

root = os.path.dirname(os.path.abspath(__file__))

def file_combined(root):

    # Define the directory where your Excel files are located
    directory = os.path.join(root,'data')

    # Get a list of all Excel files in the directory
    excel_files = [file for file in os.listdir(directory) if file.endswith(".xlsx")]

    # Create an empty list to store the dataframes
    dfs = []

    # Iterate through each Excel file
    for file in excel_files:
        # Construct the file path
        file_path = os.path.join(directory, file)
        
        # Read the Excel file into a dataframe
        df = pd.read_excel(file_path)
        df.iloc[0] = df.iloc[0].astype(str)
        df = df.reset_index(drop=True)
        df = df.rename(columns=df.iloc[0]).drop(df.index[0])
        
        
        # Append the dataframe to the list
        dfs.append(df)

    # Concatenate all dataframes into a single dataframe
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(os.path.join(root,'data_modified','combined.csv'),index=False)
    list_combined_df = [x for x in combined_df.columns if str(x) != 'nan']
    combined_df_final = combined_df[list_combined_df]
    combined_df_final.to_csv(os.path.join(root,'data_modified','combined_df_final.csv'),index=False)
    combined_df_final = combined_df[list_combined_df].drop_duplicates().reset_index(drop=True)

    return combined_df_final

def main():
    combined_df_final = file_combined(root)
    print(combined_df_final)

    # Step 1: Load the Excel file using pandas
    df = combined_df_final

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