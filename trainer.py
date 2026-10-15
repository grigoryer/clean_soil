import glob
import tomllib

import chess
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from generator import data_decoder

PIECE_MAP = [4, 3, 2, 1, 0, 5]  # python-chess pawn..king -> engine order (queen, rook, bishop, knight, pawn, king)
QA, QB = 255, 64

REC_DTYPE = np.dtype([
    ("occ",    "<u8"),          # 8 bytes, occupancy bitboard
    ("packed", "u1", (16,)),    # 16 bytes, piece nibbles
    ("stm",    "u1"),           # 1 byte
    ("score",  "<i2"),          # 2 bytes
    ("result", "u1"),           # 1 byte
    ("pad",    "u1", (4,)),     # 4 bytes of padding
])


def load_data(config):
    files = sorted(glob.glob(config["dataset"]["path"] + config["dataset"]["set"] + "*.bin"))
    data = np.concatenate([np.fromfile(p, dtype=REC_DTYPE) for p in files])
    np.random.default_rng(seed=42).shuffle(data)
    return data


def create_position(fen_string: str) -> np.ndarray:
    board = chess.Board(fen_string)
    us = board.turn

    x = np.zeros(768, dtype=np.float32)

    for sq, piece in board.piece_map().items():
        rel_color = 0 if piece.color == us else 1        # 0 = side to move's piece, 1 = opponent's
        rel_sq = sq if us == chess.WHITE else sq ^ 56     # flip the board when black is to move
        x[rel_color * 384 + PIECE_MAP[piece.piece_type - 1] * 64 + rel_sq] = 1.0

    return x


class NnueDataset(Dataset):
    def __init__(self, data):
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        fen_string, score = data_decoder(self.dataset[idx])
        position = create_position(fen_string)

        score = float(score)
        if fen_string.split()[1] == "b":  # stored score is white POV, features are side-to-move relative
            score = -score

        return position, np.float32(score)


class NnueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=768, out_features=1024),
            nn.Hardtanh(min_val=0.0, max_val=1.0),  # clipped ReLU, matches the clamp to [0, QA] in the engine
            nn.Linear(in_features=1024, out_features=1),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X).squeeze(1)
        loss = loss_fn(torch.sigmoid(pred), torch.sigmoid(y / 400))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * dataloader.batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X).squeeze(1)
            test_loss += loss_fn(torch.sigmoid(pred), torch.sigmoid(y / 400)).item()

    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")


def q16(a: np.ndarray, scale: int) -> np.ndarray:
    q = np.round(a * scale)
    if np.abs(q).max() > 32767:
        print("warning: values clipped, scale is too big for these weights")
    return np.clip(q, -32768, 32767).astype("<i2")


def export_net(model: nn.Module, path: str = "net.bin") -> None:
    """Writes the quantized net in the layout Nnue::InitNet reads."""
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    w1 = sd["linear_relu_stack.0.weight"].T                 # [1024][768] -> [768][1024]
    b1 = sd["linear_relu_stack.0.bias"]
    w2 = sd["linear_relu_stack.2.weight"].flatten()         # [1][1024] -> [1024]
    b2 = sd["linear_relu_stack.2.bias"]

    with open(path, "wb") as f:
        f.write(q16(w1, QA).tobytes())                          # accum_weights
        f.write(q16(b1, QA).tobytes())                          # accum_biases
        f.write(q16(w2, QB).tobytes())                          # output_weights
        f.write(np.round(b2 * QA * QB).astype("<i4").tobytes()) # output_bias (int32)

    print(f"Exported {path}")


def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    with open("./config/train_config.toml", "rb") as f:
        config = tomllib.load(f)

    learning_rate = 1e-3
    batch_size = 4096
    epochs = 15

    data = load_data(config)
    n_val = len(data) // 100  # hold out 1%

    train_loader = DataLoader(NnueDataset(data[:-n_val]), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(NnueDataset(data[-n_val:]), batch_size=batch_size)

    model = NnueNet().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, device)
        test_loop(val_loader, model, loss_fn, device)
    print("Done!")

    torch.save(model.state_dict(), "nnue_weights.pth")
    export_net(model, "net.bin")


if __name__ == "__main__":
    main()