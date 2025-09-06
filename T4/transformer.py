import math, os, torch, torch.nn as nn, torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

PAD_VAL = -1.0        
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 3
FFN_DIM = 256
DROPOUT = 0.1

parcels = gpd.read_file(r"C:\Users\royba\Downloads\stage\T4\dusapvhqgpj_-1097366394650450501\Parcel_BBD_Beirutmap.shp")
buildings = gpd.read_file(r"C:\Users\royba\Downloads\stage\T4\Buildings_-3411144284632197355\Beirut_Buildings.shp")

parcels = parcels.to_crs(buildings.crs)

joined = gpd.sjoin(buildings, parcels, predicate="within")

building_counts = joined.groupby("index_right").size()
single_building_parcels = building_counts[building_counts == 1].index

max_points_parcel = 0
idz=0
for parcel_idx, count in building_counts.items():
    if count != 1:
        continue  

    geom = parcels.loc[parcel_idx].geometry

    if geom is None or parcel_idx == 11217:
        continue
    if geom.geom_type == "Polygon":
        points = len(geom.exterior.coords)
    elif geom.geom_type == "MultiPolygon":
        polys=list(geom.geoms)
        for poly in polys:
            points =  len(poly.exterior.coords)
    if points > max_points_parcel:
        max_points_parcel = points
        idz = parcel_idx
print("Max points in parcels:", max_points_parcel)
print("Parcel with max points index:", idz)

max_points_building = 0
for parcel_idx, count in building_counts.items():
    if count != 1:
        continue  
    building = joined[joined.index_right == parcel_idx].geometry.iloc[0]
    if building is None:
        continue
    if building.geom_type == "Polygon":
        points = len(building.exterior.coords)
    elif building.geom_type == "MultiPolygon":
        polys=list(building.geoms)
        points = max(len(poly.exterior.coords) for poly in polys)
    else:
        continue

    if points > max_points_building:
        max_points_building = points
        max_index = parcel_idx

building = joined[joined.index_right == max_index].geometry.iloc[0]

print("Building with max points index:", max_index)
print("Max points in buildings:", max_points_building)


MAX_POINTS_PARCEL = max_points_parcel
MAX_POINTS_BUILDING = max_points_building
MAX_POINTS = max(MAX_POINTS_PARCEL, MAX_POINTS_BUILDING)


def polygon_to_coords(poly):
    """Extract raw polygon coords as numpy array (without padding)."""
    return np.array(poly.exterior.coords)[:, :2]


def normalize(parcel_coords, building_coords):
    min_vals = parcel_coords.min(axis=0)
    max_vals = parcel_coords.max(axis=0)
    parcel_norm = (parcel_coords - min_vals) / (max_vals - min_vals + 1e-9)
    building_norm = (building_coords - min_vals) / (max_vals - min_vals + 1e-9)

    return parcel_norm, building_norm, (min_vals, max_vals)


def denormalize(parcel_coords_norm, building_coords_norm, bounds):
    min_vals, max_vals = bounds
    parcel_coords = parcel_coords_norm * (max_vals - min_vals) + min_vals
    building_coords = building_coords_norm * (max_vals - min_vals) + min_vals

    return parcel_coords, building_coords

X_norm, Y_norm = [], []
bounds = []

for idx in single_building_parcels:
    parcel = parcels.loc[[idx]].geometry.iloc[0]
    building = joined[joined.index_right == idx].geometry.iloc[0]

    if parcel is None or building is None:
        continue
    if parcel.geom_type != "Polygon" or building.geom_type != "Polygon" or idx == 11217:
        continue

    parcel_coords = polygon_to_coords(parcel)
    building_coords = polygon_to_coords(building)
    parcel_norm, building_norm, pbounds = normalize(parcel_coords, building_coords)
    if idx == 9:
        plt.figure(figsize=(6,6))
        plt.title("Parcel+blg")
        plt.plot(parcel_coords[:, 0], parcel_coords[:, 1], "b.-", label="Parcel")
        plt.plot(building_coords[:, 0], building_coords[:, 1], "r.-", label="Building")
        plt.legend()
        plt.show()

        plt.figure(figsize=(6,6))
        plt.title("Parcel+blg (normalized)")
        plt.plot(parcel_norm[:, 0], parcel_norm[:, 1], "b.-", label="Parcel")
        plt.plot(building_norm[:, 0], building_norm[:, 1], "r.-", label="Building")
        plt.legend()
        plt.show()

    X_norm.append(parcel_norm)
    Y_norm.append(building_norm)
    bounds.append(pbounds)


class PolyDataset(Dataset):
    def __init__(self, X_list, Y_list):
        assert len(X_list) == len(Y_list)
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X_list]
        self.Y = [torch.tensor(y, dtype=torch.float32) for y in Y_list]
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]

def collate_polys(batch):
    xs, ys = zip(*batch)
    B = len(xs)
    Ls = [x.shape[0] for x in xs]
    Lt = [y.shape[0] for y in ys]
    Ls_max, Lt_max = max(Ls), max(Lt)

    def pad_to(seq_list, Lmax):
        out = []
        for s in seq_list:
            pad = Lmax - s.shape[0]
            if pad > 0:
                s = torch.cat([s, torch.full((pad, 2), PAD_VAL)], dim=0)
            out.append(s)
        out = torch.stack(out, dim=0)  
        key_mask = (out[:, :, 0] == PAD_VAL) & (out[:, :, 1] == PAD_VAL)
        return out, key_mask

    src, src_key_padding_mask = pad_to(xs, Ls_max)
    tgt, tgt_key_padding_mask = pad_to(ys, Lt_max)

    BOS = torch.zeros((B, 1, 2), dtype=torch.float32)
    tgt_in = torch.cat([BOS, tgt[:, :-1, :]], dim=1)    

    return (src, src_key_padding_mask), (tgt_in, tgt, tgt_key_padding_mask)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0., max_len=2048):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)             
        position = torch.arange(0, max_len).unsqueeze(1)    
        div_term = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        L = x.size(0)
        x = x + self.pe[:L].unsqueeze(1)                
        return self.dropout(x)


class CoordTransformer(nn.Module):
    def __init__(self, d_model=D_MODEL, nhead=NHEAD,
                 num_layers=NUM_LAYERS, ffn_dim=FFN_DIM, dropout=DROPOUT):
        super().__init__()
        self.in_proj  = nn.Linear(2, d_model)
        self.out_proj = nn.Linear(d_model, 2)

        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ffn_dim, dropout, batch_first=False)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, ffn_dim, dropout, batch_first=False)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.pos_dec = PositionalEncoding(d_model, dropout)

    def forward(self, src_xy, tgt_in_xy, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src = self.in_proj(src_xy)        
        tgt = self.in_proj(tgt_in_xy)       

        src = src.transpose(0,1)        
        tgt = tgt.transpose(0,1)            
        src = self.pos_enc(src)
        tgt = self.pos_dec(tgt)

        Lt = tgt.size(0)
        causal_mask = torch.triu(torch.ones(Lt, Lt, device=tgt.device) == 1, diagonal=1)

        mem = self.encoder(src, src_key_padding_mask=src_key_padding_mask) 
        out = self.decoder(tgt, mem,
                           tgt_mask=causal_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=src_key_padding_mask)    

        out = out.transpose(0,1)         
        out = self.out_proj(out)        
        return out


def masked_mse(pred, target, key_padding_mask):
    mask = ~key_padding_mask   
    mask = mask.unsqueeze(-1).float()
    diff2 = (pred - target)**2 * mask
    denom = mask.sum().clamp_min(1.0)
    return diff2.sum() / denom

def edge_loss(pred, target, key_padding_mask):
    mask = ~key_padding_mask
    pred_e = pred[:, 1:, :] - pred[:, :-1, :]
    true_e = target[:, 1:, :] - target[:, :-1, :]
    valid = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1).float()
    return ((pred_e - true_e)**2 * valid).sum() / valid.sum().clamp_min(1.0)


def train_transformer(dataset, save_path="parcel2building_transformer.pth",
                      batch_size=16, lr=1e-3, num_epochs=60, patience=10,
                      resume=True, device=None, add_edge_loss=True):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n = len(dataset)
    val_n = max(1, int(0.2*n))
    train_n = n - val_n
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_n, val_n])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_polys)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_polys)

    model = CoordTransformer().to(device)

    if resume and os.path.exists(save_path):
        print(f"Resuming from {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))

    opt = optim.Adam(model.parameters(), lr=lr)
    sched = lr_scheduler.StepLR(opt, step_size=12, gamma=0.5)

    best_val, bad = float("inf"), 0

    for epoch in range(1, num_epochs+1):
        model.train()
        tr_loss = 0.0
        for (src, src_pad), (tgt_in, tgt_out, tgt_pad) in train_loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            src_pad, tgt_pad = src_pad.to(device), tgt_pad.to(device)

            opt.zero_grad()
            pred = model(src, tgt_in, src_key_padding_mask=src_pad, tgt_key_padding_mask=tgt_pad)
            loss = masked_mse(pred, tgt_out, tgt_pad)
            if add_edge_loss:
                loss += 0.05 * edge_loss(pred, tgt_out, tgt_pad)
            loss.backward()
            opt.step()
            tr_loss += loss.item()

        tr_loss /= len(train_loader)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for (src, src_pad), (tgt_in, tgt_out, tgt_pad) in val_loader:
                src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
                src_pad, tgt_pad = src_pad.to(device), tgt_pad.to(device)
                pred = model(src, tgt_in, src_key_padding_mask=src_pad, tgt_key_padding_mask=tgt_pad)
                loss = masked_mse(pred, tgt_out, tgt_pad)
                if add_edge_loss:
                    loss += 0.05 * edge_loss(pred, tgt_out, tgt_pad)
                va_loss += loss.item()
        va_loss /= len(val_loader)

        print(f"Epoch {epoch:02d}/{num_epochs} | Train {tr_loss:.4f} | Val {va_loss:.4f}")

        if va_loss < best_val - 1e-5:
            best_val, bad = va_loss, 0
            torch.save(model.state_dict(), save_path)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

        sched.step()

    print("Best val:", best_val, " | saved to:", save_path)
    return model


dataset = PolyDataset(X_norm, Y_norm)

model = train_transformer(
    dataset,
    save_path="parcel2building_transformer.pth",
    batch_size=16, lr=1e-3, num_epochs=0, patience=12, resume=True
)

def greedy_decode(model, src_xy, max_len=256, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        src = torch.tensor(src_xy, dtype=torch.float32).unsqueeze(0).to(device) 
        src_pad = (src[:, :, 0] == PAD_VAL) & (src[:, :, 1] == PAD_VAL)

        BOS = torch.zeros((1, 1, 2), dtype=torch.float32, device=device)
        tgt = BOS
        for _ in range(max_len-1):
            tgt_pad = (tgt[:, :, 0] == PAD_VAL) & (tgt[:, :, 1] == PAD_VAL)
            out = model(src, tgt, src_key_padding_mask=src_pad, tgt_key_padding_mask=tgt_pad)
            next_xy = out[:, -1:, :]
            tgt = torch.cat([tgt, next_xy], dim=1)

        return tgt.squeeze(0).cpu().numpy()
    
def plot_prediction(parcel, true_building, pred_building, title="Prediction"):
    plt.figure(figsize=(5,5))
    plt.plot(parcel[:,0], parcel[:,1], "b-o", label="Parcel")
    if true_building is not None:
        plt.plot(true_building[:,0], true_building[:,1], "g-o", label="True Building")
    plt.plot(pred_building[:,0], pred_building[:,1], "r-o", label="Predicted Building")
    plt.legend()
    plt.title(title)
    plt.show()


idx = 0
parcel = dataset.X[idx].numpy()
true_building = dataset.Y[idx].numpy()
pred_building = greedy_decode(model, parcel, max_len=parcel.shape[0]+5)

plot_prediction(parcel, true_building, pred_building, title=f"Sample {idx}")