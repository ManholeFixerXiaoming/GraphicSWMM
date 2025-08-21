import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import pandas as pd
import torch
from torch import nn, optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt

batch_size = 1

# Load edge data
edges = np.load(r'/hy-tmp/cqm_data/edge_index_new.npy')
edges_weight = np.load(r'/hy-tmp/cqm_data/edge_weight_new.npy')
x_combine_final = np.load(r'/hy-tmp/cqm_data/x_new.npy')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load node feature matrix
x_combine = torch.tensor(x_combine_final, dtype=torch.float).to(device)

# Convert edge info to torch_geometric format
edge_index = torch.tensor(edges, dtype=torch.long).to(device)
print(f"edge_index shape (after transpose): {edge_index.shape}")
edge_attr = torch.tensor(edges_weight, dtype=torch.float).to(device)

# Set random seed for reproducibility
seed_value = 1314
np.random.seed(seed_value)
torch.manual_seed(seed_value)


##########################################
# Custom Dataset
class RainfallDataset(Dataset):
    def __init__(self, x_data, y1_data, y2_data, y3_data):
        # Initialize dataset
        self.x_data = x_data
        self.y1_data = y1_data
        self.y2_data = y2_data
        self.y3_data = y3_data

    def __len__(self):
        # Return dataset size
        return len(self.x_data)

    def __getitem__(self, idx):
        # Return one sample
        x = torch.tensor(self.x_data[idx], dtype=torch.float32)
        y1 = torch.tensor(self.y1_data[idx], dtype=torch.float32)
        y2 = torch.tensor(self.y2_data[idx], dtype=torch.float32)
        y3 = torch.tensor(self.y3_data[idx], dtype=torch.float32)
        return x, y1.squeeze(), y2.squeeze(), y3.squeeze()


##########################################
# Load dataset
x_path = r'/hy-tmp/cqm_data/12000input.npy'
y2_path = r'/hy-tmp/cqm_data/data_Node_outflow.npy'
y1_path = r'/hy-tmp/cqm_data/data_link_outflow.npy'
x_data = np.load(x_path)
y1_data = np.load(y1_path)
y2_data = np.load(y2_path)

# Extra training samples
x_path1000 = r'/hy-tmp/cqm_data/1000/sp1000.npy'
y1_path1000 = r'/hy-tmp/cqm_data/1000/data_link_outflow.npy'
y2_path1000 = r'/hy-tmp/cqm_data/1000/data_Node_outflow.npy'

x_data1000 = np.load(x_path1000)
y1_data1000 = np.load(y1_path1000)
y2_data1000 = np.load(y2_path1000)
x_data = np.concatenate((x_data, x_data1000), axis=0)
y1_data = np.concatenate((y1_data, y1_data1000), axis=0)
y2_data = np.concatenate((y2_data, y2_data1000), axis=0)

y2_data[:, :, 0] = 0
x_data = x_data.reshape(x_data.shape[0], 1, 65 * 3)  # Reshape for embedding

# Process y3_data
y3_data = y2_data
y2_sum = np.sum(y2_data, axis=(1, 2))
y2_sum = np.clip(y2_sum, None, 992.052)
y2_data = y2_sum
accc = np.load(r'/hy-tmp/cqm_data/new_cl.npy')
num_samples, num_classes, seq_length = y3_data.shape[0], len(np.unique(accc[:, 0])), y3_data.shape[2]
y4_data = np.zeros((num_samples, num_classes, seq_length))

# Aggregate by clusters
for class_type in range(num_classes):
    points_in_class = accc[accc[:, 0] == class_type, 1]
    y4_data[:, class_type, :] = y3_data[:, points_in_class, :].sum(axis=1)
y3_data = y4_data


##########################################
# Train/Validation/Test split
total_size = len(x_data)
train_size = 9000
val_size = 2000
test_size = total_size - train_size - val_size

x_train, x_rem, y1_train, y1_rem, y2_train, y2_rem, y3_train, y3_rem = train_test_split(
    x_data, y1_data, y2_data, y3_data, train_size=train_size, random_state=9512, shuffle=True
)

x_val, x_test, y1_val, y1_test, y2_val, y2_test, y3_val, y3_test = train_test_split(
    x_rem, y1_rem, y2_rem, y3_rem, test_size=test_size, random_state=9512, shuffle=True
)

##########################################
# Create Datasets and DataLoaders
batch_size = 1

train_dataset = RainfallDataset(x_train, y1_train, y2_train, y3_train)
val_dataset = RainfallDataset(x_val, y1_val, y2_val, y3_val)
test_dataset = RainfallDataset(x_test, y1_test, y2_test, y3_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

print(f'Train dataset size: {len(train_dataset)}')
print(f'Validation dataset size: {len(val_dataset)}')
print(f'Test dataset size: {len(test_dataset)}')


# Calculate R²
def calculate_r2(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    return r2_score(y_true, y_pred)


from torch.nn import BatchNorm1d


class GCNNodeEdge(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels_node, out_channels_edge, rain_length, lstm_hidden=121,
                 point_num=35):
        super(GCNNodeEdge, self).__init__()
        # Define GCN layers
        self.Gconv1 = GCNConv(in_channels, hidden_channels)
        self.Gconvlid = torch.nn.Linear(65 * 3, hidden_channels)
        self.Gconv2 = GCNConv(hidden_channels, 2 * hidden_channels)
        self.Gconv3 = GCNConv(2 * hidden_channels, 1 * hidden_channels)
        self.Gconv4 = GCNConv(1 * hidden_channels, 24)

        # Batch Normalization layers
        self.bn1 = BatchNorm1d(hidden_channels)
        self.bnlid = BatchNorm1d(hidden_channels)
        self.bn2 = BatchNorm1d(2 * hidden_channels)
        self.bn3 = BatchNorm1d(1 * hidden_channels)
        self.bn4 = BatchNorm1d(out_channels_node)

    def forward(self, x_in, edge_index, edge_weight, lid_timeseries, point_num=988):
        edge_weight = edge_weight.squeeze()
        x = self.Gconv1(x_in.squeeze(), edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        lid_timeseries = self.Gconvlid(lid_timeseries.squeeze())
        lid_timeseries = F.relu(lid_timeseries)
        x = x + lid_timeseries
        mask = (x_in[:, 0] == 0)
        x = x * (~mask).unsqueeze(1)
        x = self.Gconv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.Gconv3(x, edge_index, edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.Gconv4(x, edge_index, edge_weight)
        x = self.bn4(x)
        x = F.relu(x)
        sumx = torch.sum(x)

        return x, sumx


net = GCNNodeEdge(in_channels=9, hidden_channels=128 * 2, out_channels_node=24, out_channels_edge=1, rain_length=121)

# Loss, optimizer, scheduler
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

# Metrics
val_r2_scores_y2 = []
test_r2_scores_y2 = []

# Training loop
epochs = 200
train_losses = []
val_losses = []
validate_every = 4

net.to(device)

val_diff_aver_list = []
val_mean_relative_error_list = []
test_diff_aver_list = []
test_mean_relative_error_list = []

for epoch in range(epochs):
    net.train()
    running_loss = 0.0

    for i, (x_batch, y1_batch, y2_batch, y3_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
        x_batch, y1_batch, y2_batch, y3_batch = x_batch.to(device), y1_batch.to(device), y2_batch.to(device), y3_batch.to(device)

        optimizer.zero_grad()

        outputs = net(x_combine, edge_index, edge_attr, x_batch)
        y3_pred, y2_pred = outputs
        y2_batch = y2_batch.squeeze()
        loss1 = criterion(y3_pred, y3_batch)
        loss2 = criterion(y2_pred, y2_batch)
        loss = loss2 + loss1

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}')
    train_losses.append(running_loss / len(train_loader))

    if (epoch + 1) % validate_every == 0:
        val_loss = 0.0
        all_y2_val = []
        all_y2_val_pred = []
        all_diff_val = []
        with torch.no_grad():
            for x_val, y1_val, y2_val, y3_val in tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}"):
                x_val, y1_val, y2_val, y3_val = x_val.to(device), y1_val.to(device), y2_val.to(device), y3_val.to(device)

                outputs = net(x_combine, edge_index, edge_attr, x_val)
                y3_val_pred, y2_val_pred = outputs
                y3_val = y3_val.squeeze()
                y2_val = y2_val.squeeze()
                val_loss3 = criterion(y3_val_pred, y3_val)
                val_loss2 = criterion(y2_val_pred, y2_val)
                val_loss = val_loss2 + val_loss3
                val_loss += val_loss.item()

                all_y2_val.append(y2_val.item())
                all_y2_val_pred.append(y2_val_pred.item())
                diff = y2_val_pred.item() - y2_val.item()
                all_diff_val.append(diff)

        val_r2_y2 = r2_score(all_y2_val, all_y2_val_pred)
        val_losses.append(val_loss / len(val_loader))
        val_r2_scores_y2.append(val_r2_y2)
        val_diff_aver = sum(abs(x) for x in all_diff_val) / len(all_diff_val)
        val_diff_aver_list.append(val_diff_aver)
        all_y2_val = np.array(all_y2_val)
        all_y2_val_pred = np.array(all_y2_val_pred)

        mean_relative_error_val = np.mean(np.abs((all_y2_val - all_y2_val_pred) / all_y2_val))
        val_mean_relative_error_list.append(mean_relative_error_val)
        print(f'Validation Loss: {val_loss / len(val_loader)}, R² for y2: {val_r2_y2}, val_diff_aver: {val_diff_aver}, mean_relative_error: {mean_relative_error_val}')
    scheduler.step()

print('Training finished')

# Test
test_loss = 0.0
all_y2_test = []
all_y2_test_pred = []
all_diff_test = []
with torch.no_grad():
    for x_test, y1_test, y2_test, y3_test in tqdm(test_loader, desc="Evaluating on test set"):
        x_test, y1_test, y2_test, y3_test = x_test.to(device), y1_test.to(device), y2_test.to(device), y3_test.to(device)

        outputs = net(x_combine, edge_index, edge_attr, x_test)
        y3_test_pred, y2_test_pred = outputs
        test_loss3 = criterion(y3_test_pred, y3_test)
        test_loss2 = criterion(y2_test_pred, y2_test)
        test_lossAll = test_loss2 + test_loss3

        test_loss += test_lossAll.item()

        all_y2_test.append(y2_test.item())
        all_y2_test_pred.append(y2_test_pred.item())
        diff = y2_test_pred.item() - y2_test.item()
        all_diff_test.append(diff)

all_y2_test = np.array(all_y2_test)
all_y2_test_pred = np.array(all_y2_test_pred)

mean_relative_error_test = np.mean(np.abs((all_y2_test - all_y2_test_pred) / all_y2_test))
test_mean_relative_error_list.append(mean_relative_error_test)
test_r2_y2 = r2_score(all_y2_test, all_y2_test_pred)
test_r2_scores_y2.append(test_r2_y2)
diff_aver = sum(abs(x) for x in all_diff_test) / len(all_diff_test)
test_diff_aver_list.append(diff_aver)
print(f'Test Loss: {test_loss / len(test_loader)}, R² for y2: {test_r2_y2}, diff average: {diff_aver}, mean_relative_error_test: {mean_relative_error_test}')

# Save model
checkpoint = {
    "net": net.state_dict(),
    'optimizer': optimizer.state_dict(),
    "epoch": epochs
}
torch.save(checkpoint, 'FC_embed_light.pth')

# Save metrics to Excel
data = {
    'Train Losses': train_losses,
    'Validation Losses': val_losses,
    'Validation Diff Average': val_diff_aver_list,
    'Validation Mean Relative Error': val_mean_relative_error_list
}
max_length = max(len(val_r2_scores_y2), len(test_r2_scores_y2), len(test_diff_aver_list), len(test_mean_relative_error_list))

val_r2_scores_y2 += [None] * (max_length - len(val_r2_scores_y2))
test_r2_scores_y2 += [None] * (max_length - len(test_r2_scores_y2))
test_diff_aver_list += [None] * (max_length - len(test_diff_aver_list))
test_mean_relative_error_list += [None] * (max_length - len(test_mean_relative_error_list))

r2_data = {
    'Validation R² (y2)': val_r2_scores_y2,
    'Test R² (y2)': test_r2_scores_y2,
    'Test Diff Average': test_diff_aver_list,
    'Test Mean Relative Error': test_mean_relative_error_list
}
df_r2 = pd.DataFrame(r2_data)

df_losses = pd.DataFrame.from_dict(data, orient='index')

excel_path = r'FC_embed_light.xlsx'
with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
    df_losses.to_excel(writer, sheet_name='Losses', header=False)
    df_r2.to_excel(writer, sheet_name='R² Scores')

print('R² and losses saved to Excel')
