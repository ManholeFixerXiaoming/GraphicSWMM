from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GCNConv
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SCRIPT_DIR = Path(__file__).resolve().parent
PACK_ROOT = SCRIPT_DIR.parent
RAW_GRAPH_DIR = PACK_ROOT / "clustering" / "raw_graph"
DATA_DIR = PACK_ROOT / "labels"
NODE_LID_PATH = SCRIPT_DIR / "node_lid_inputs.npy"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs_node1020"
SEED = 1314


class NodeLidDataset(Dataset):
    def __init__(self, x_data: np.ndarray, y_total: np.ndarray, y_node: np.ndarray):
        self.x_data = x_data.astype(np.float32)
        self.y_total = y_total.astype(np.float32)
        self.y_node = y_node.astype(np.float32)

    def __len__(self) -> int:
        return self.x_data.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.x_data[idx], dtype=torch.float32),
            torch.tensor(self.y_total[idx], dtype=torch.float32),
            torch.tensor(self.y_node[idx], dtype=torch.float32),
        )


class GCNNode1020Dual(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, time_steps: int):
        super().__init__()
        self.gconv1 = GCNConv(in_channels, hidden_channels)
        self.lid_embed = nn.Linear(3, hidden_channels)
        self.gconv2 = GCNConv(hidden_channels, hidden_channels * 2)
        self.gconv3 = GCNConv(hidden_channels * 2, hidden_channels)
        self.gconv4 = GCNConv(hidden_channels, time_steps)
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.ln_lid = nn.LayerNorm(hidden_channels)
        self.ln2 = nn.LayerNorm(hidden_channels * 2)
        self.ln3 = nn.LayerNorm(hidden_channels)

    def forward(self, x_graph: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, node_lid: torch.Tensor):
        if node_lid.ndim == 3:
            node_lid = node_lid.squeeze(0)
        edge_weight = edge_weight.squeeze()
        x = torch.tanh(self.ln1(self.gconv1(x_graph, edge_index, edge_weight)))
        lid_feature = F.relu(self.ln_lid(self.lid_embed(node_lid)))
        x = x + lid_feature
        x = torch.tanh(self.ln2(self.gconv2(x, edge_index, edge_weight)))
        x = torch.tanh(self.ln3(self.gconv3(x, edge_index, edge_weight)))
        node_series = self.gconv4(x, edge_index, edge_weight)
        total_value = torch.sum(node_series)
        return node_series, total_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--node-lid", default=str(NODE_LID_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing training file: {path}")
    return path


def load_graph(device: torch.device):
    x_graph = np.load(require_file(RAW_GRAPH_DIR / "X_combine_final.npy")).astype(np.float32)
    edges = np.load(require_file(RAW_GRAPH_DIR / "edges_all_links.npy")).astype(np.int64) - 1
    edge_weight = np.load(require_file(RAW_GRAPH_DIR / "edges_weight_all_links.npy")).astype(np.float32)
    return (
        torch.tensor(x_graph, dtype=torch.float32, device=device),
        torch.tensor(edges.T, dtype=torch.long, device=device),
        torch.tensor(edge_weight, dtype=torch.float32, device=device),
        x_graph,
    )


def load_training_data(data_dir: Path, node_lid_path: Path):
    x_data = np.load(require_file(node_lid_path)).astype(np.float32)
    y_node = np.load(require_file(data_dir / "data_Node_inflow.npy")).astype(np.float32)
    if x_data.ndim != 3 or x_data.shape[1:] != (1020, 3):
        raise RuntimeError(f"node_lid_inputs.npy should be [N, 1020, 3], got {x_data.shape}")
    if y_node.shape[0] != x_data.shape[0] or y_node.shape[1] != x_data.shape[1]:
        raise RuntimeError(f"sample or node mismatch: x={x_data.shape}, y={y_node.shape}")
    y_total = y_node.sum(axis=(1, 2)).astype(np.float32)
    return x_data, y_total, y_node


def make_loaders(x_data: np.ndarray, y_total: np.ndarray, y_node: np.ndarray, train_size: int, val_size: int, batch_size: int):
    total_size = x_data.shape[0]
    test_size = total_size - train_size - val_size
    if test_size <= 0:
        raise RuntimeError("train-size + val-size must be less than the number of samples.")
    x_train, x_rem, yt_train, yt_rem, yn_train, yn_rem = train_test_split(
        x_data, y_total, y_node, train_size=train_size, random_state=9512, shuffle=True
    )
    x_val, x_test, yt_val, yt_test, yn_val, yn_test = train_test_split(
        x_rem, yt_rem, yn_rem, test_size=test_size, random_state=9512, shuffle=True
    )
    target_scales = {"node_scale": 1.0, "total_scale": 1.0}
    return (
        DataLoader(NodeLidDataset(x_train, yt_train, yn_train), batch_size=batch_size, shuffle=True),
        DataLoader(NodeLidDataset(x_val, yt_val, yn_val), batch_size=1, shuffle=False),
        DataLoader(NodeLidDataset(x_test, yt_test, yn_test), batch_size=1, shuffle=False),
        target_scales,
    )


def normalized_total_prediction(pred_scalar: torch.Tensor, target_scales: dict[str, float]) -> torch.Tensor:
    return pred_scalar


def evaluate(model, loader, x_graph, edge_index, edge_weight, criterion, device, target_scales: dict[str, float]):
    model.eval()
    losses: list[float] = []
    true_total: list[float] = []
    pred_total: list[float] = []
    with torch.no_grad():
        for x_lid, y_total, y_node in loader:
            x_lid = x_lid.to(device)
            y_total = y_total.to(device).squeeze()
            y_node = y_node.to(device).squeeze()
            pred_node, pred_scalar = model(x_graph, edge_index, edge_weight, x_lid)
            pred_total_norm = normalized_total_prediction(pred_scalar.squeeze(), target_scales)
            loss = criterion(pred_node, y_node) + criterion(pred_total_norm, y_total)
            losses.append(float(loss.item()))
            true_total.append(float(y_total.detach().cpu()))
            pred_total.append(float(pred_scalar.detach().cpu()))
    r2 = r2_score(true_total, pred_total) if len(set(true_total)) > 1 else float("nan")
    mae = float(np.mean(np.abs(np.array(true_total) - np.array(pred_total))))
    return float(np.mean(losses)), float(r2), mae


def main() -> None:
    args = parse_args()
    set_seed()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_graph, edge_index, edge_weight, x_graph_np = load_graph(device)
    x_data, y_total, y_node = load_training_data(Path(args.data_dir), Path(args.node_lid))
    train_loader, val_loader, test_loader, target_scales = make_loaders(
        x_data, y_total, y_node, args.train_size, args.val_size, args.batch_size
    )
    model = GCNNode1020Dual(
        in_channels=x_graph_np.shape[1],
        hidden_channels=args.hidden,
        time_steps=y_node.shape[2],
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for x_lid, y_total_batch, y_node_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            if x_lid.shape[0] != 1:
                raise RuntimeError("batch-size must remain 1.")
            x_lid = x_lid.to(device)
            y_total_batch = y_total_batch.to(device).squeeze()
            y_node_batch = y_node_batch.to(device).squeeze()
            optimizer.zero_grad()
            pred_node, pred_scalar = model(x_graph, edge_index, edge_weight, x_lid)
            pred_total_norm = normalized_total_prediction(pred_scalar.squeeze(), target_scales)
            loss = criterion(pred_node, y_node_batch) + criterion(pred_total_norm, y_total_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        scheduler.step()
        val_loss, val_r2, val_mae = evaluate(
            model, val_loader, x_graph, edge_index, edge_weight, criterion, device, target_scales
        )
        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "val_r2_total": val_r2,
            "val_mae_total": val_mae,
        }
        history.append(epoch_info)
        print(json.dumps(epoch_info))
    test_loss, test_r2, test_mae = evaluate(
        model, test_loader, x_graph, edge_index, edge_weight, criterion, device, target_scales
    )
    checkpoint = {
        "net": model.state_dict(),
        "config": {
            "in_channels": int(x_graph_np.shape[1]),
            "hidden_channels": int(args.hidden),
            "time_steps": int(y_node.shape[2]),
            "node_count": int(x_graph_np.shape[0]),
            "lid_input_dim_per_node": 3,
        },
        "target_scales": target_scales,
        "metrics": {
            "test_loss": test_loss,
            "test_r2_total": test_r2,
            "test_mae_total": test_mae,
        },
    }
    torch.save(checkpoint, output_dir / "node1020_gcn_dual_v3.pth")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "test_metrics.json").write_text(json.dumps(checkpoint["metrics"], indent=2), encoding="utf-8")
    print("training finished")


if __name__ == "__main__":
    main()
