from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

SCRIPT_DIR = Path(__file__).resolve().parent
PACK_ROOT = SCRIPT_DIR.parent
RAW_GRAPH_DIR = PACK_ROOT / "clustering" / "raw_graph"
CLUSTER_DIR = PACK_ROOT / "clustering" / "community_output"
MODEL_PATH = SCRIPT_DIR / "outputs_node1020" / "node1020_gcn_dual_v3.pth"
FINAL_INP = PACK_ROOT / "swmm" / "bellinge_final.inp"

model_loaded = False
net = None
x_graph = None
edge_index = None
edge_attr = None
target_scales = None
node_lid_index = None
subcatchment_categories = None
community_area_repeated = None
subcatchment_areas = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def forward(self, x_in: torch.Tensor, edge_idx: torch.Tensor, edge_weight: torch.Tensor, node_lid: torch.Tensor):
        if node_lid.ndim == 3:
            node_lid = node_lid.squeeze(0)
        edge_weight = edge_weight.squeeze()
        x = torch.tanh(self.ln1(self.gconv1(x_in, edge_idx, edge_weight)))
        lid_feature = F.relu(self.ln_lid(self.lid_embed(node_lid)))
        x = x + lid_feature
        x = torch.tanh(self.ln2(self.gconv2(x, edge_idx, edge_weight)))
        x = torch.tanh(self.ln3(self.gconv3(x, edge_idx, edge_weight)))
        node_series = self.gconv4(x, edge_idx, edge_weight)
        total_value = torch.sum(node_series)
        return node_series, total_value


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing inference file: {path}")
    return path


def read_subcatchment_areas(inp_path: Path) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    areas: list[float] = []
    in_subcatchments = False
    for raw_line in inp_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_subcatchments = line.upper() == "[SUBCATCHMENTS]"
            continue
        if not in_subcatchments or line.startswith(";"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            names.append(parts[0])
            areas.append(float(parts[3]))
    if not names:
        raise RuntimeError(f"failed to read [SUBCATCHMENTS] from {inp_path}")
    return names, np.array(areas, dtype=np.float32)


def load_model_and_data() -> None:
    global model_loaded, net, x_graph, edge_index, edge_attr, target_scales
    global node_lid_index, subcatchment_categories, community_area_repeated, subcatchment_areas
    if model_loaded:
        return
    checkpoint = torch.load(require_file(MODEL_PATH), map_location=device)
    config = checkpoint["config"]
    net = GCNNode1020Dual(
        in_channels=int(config["in_channels"]),
        hidden_channels=int(config["hidden_channels"]),
        time_steps=int(config["time_steps"]),
    ).to(device)
    net.load_state_dict(checkpoint["net"])
    net.eval()
    target_scales = checkpoint.get("target_scales", {"node_scale": 1.0, "total_scale": 1.0})
    x_graph_np = np.load(require_file(RAW_GRAPH_DIR / "X_combine_final.npy")).astype(np.float32)
    edges = np.load(require_file(RAW_GRAPH_DIR / "edges_all_links.npy")).astype(np.int64) - 1
    edge_weight_np = np.load(require_file(RAW_GRAPH_DIR / "edges_weight_all_links.npy")).astype(np.float32)
    x_graph = torch.tensor(x_graph_np, dtype=torch.float32, device=device)
    edge_index = torch.tensor(edges.T, dtype=torch.long, device=device)
    edge_attr = torch.tensor(edge_weight_np, dtype=torch.float32, device=device)
    node_lid_index = np.load(require_file(RAW_GRAPH_DIR / "NodeLidIndex.npy")).astype(int).reshape(-1)
    saved_subcatchment_names = np.load(require_file(RAW_GRAPH_DIR / "subcatchment_names.npy"), allow_pickle=True).astype(str)
    inp_subcatchment_names, subcatchment_areas = read_subcatchment_areas(require_file(FINAL_INP))
    if list(saved_subcatchment_names) != inp_subcatchment_names:
        raise RuntimeError("subcatchment_names.npy order does not match INP [SUBCATCHMENTS].")
    subcatchment_categories = np.load(require_file(CLUSTER_DIR / "subcatchment_community.npy")).astype(int).reshape(-1)
    community_area = np.load(require_file(CLUSTER_DIR / "community_total_area.npy")).astype(float).reshape(-1)
    community_area_repeated = np.repeat(community_area, 3)
    model_loaded = True


def community_lid_to_node_lid(lid_series) -> np.ndarray:
    load_model_and_data()
    lid_array = np.asarray(lid_series, dtype=float).reshape(-1)
    if lid_array.size != 47 * 3:
        raise ValueError(f"LID decision vector must have 141 values, got {lid_array.size}")
    assert community_area_repeated is not None
    lid_ratio = np.divide(
        lid_array,
        community_area_repeated,
        out=np.zeros_like(lid_array, dtype=float),
        where=community_area_repeated > 0,
    ).reshape(47, 3)
    assert node_lid_index is not None and subcatchment_categories is not None and subcatchment_areas is not None
    node_lid = np.zeros((1020, 3), dtype=np.float32)
    for sub_idx, community_id_1based in enumerate(subcatchment_categories):
        node_idx = int(node_lid_index[sub_idx]) - 1
        community_idx = int(community_id_1based) - 1
        node_lid[node_idx, :] += lid_ratio[community_idx, :] * subcatchment_areas[sub_idx]
    return node_lid


def run_gcn_model(lid_series) -> float:
    load_model_and_data()
    node_lid = community_lid_to_node_lid(lid_series)
    node_lid_tensor = torch.tensor(node_lid, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, total_value = net(x_graph, edge_index, edge_attr, node_lid_tensor)
    return float(total_value.detach().cpu().item())


def run_gcn_model_series(lid_series) -> np.ndarray:
    load_model_and_data()
    node_lid = community_lid_to_node_lid(lid_series)
    node_lid_tensor = torch.tensor(node_lid, dtype=torch.float32, device=device)
    with torch.no_grad():
        node_series, _ = net(x_graph, edge_index, edge_attr, node_lid_tensor)
    return node_series.detach().cpu().numpy()
