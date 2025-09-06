from shapely.geometry import Polygon
import geopandas as gpd
import os
import ezdxf
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader , random_split
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Train a model on parcel data")
parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer")
args = parser.parse_args()
num_epochs = args.num_epochs

parcels = gpd.read_file(r"C:\Users\royba\Downloads\stage\T4\dusapvhqgpj_-1097366394650450501\Parcel_BBD_Beirutmap.shp")
buildings = gpd.read_file(r"C:\Users\royba\Downloads\stage\T4\Buildings_-3411144284632197355\Beirut_Buildings.shp")

parcels = parcels.to_crs(buildings.crs)


os.makedirs("output/parcels_only", exist_ok=True)
os.makedirs("output/parcels_with_buildings", exist_ok=True)

joined = gpd.sjoin(buildings, parcels, predicate="within")

building_counts = joined.groupby("index_right").size()
single_building_parcels = building_counts[building_counts == 1].index

for parcel_idx, count in building_counts.items():
    if count != 1:
        continue  
    
    parcel = parcels.loc[[parcel_idx]]
    building = joined[joined.index_right == parcel_idx]


    parcel_only_path = f"output/parcels_only/parcel_{parcel_idx}.png"
    parcel_with_building_path = f"output/parcels_with_buildings/parcel_{parcel_idx}_with_building.png"
    
    if not os.path.exists(parcel_only_path):
        fig, ax = plt.subplots(figsize=(6,6))
        gpd.GeoDataFrame([parcel], crs=parcels.crs).plot(ax=ax, facecolor="none", edgecolor="blue", linewidth=2)
        plt.axis("equal")
        plt.savefig(parcel_only_path, dpi=300)
        plt.close()
    #else:
        #print(f"⏩ Skipping {parcel_only_path}, already exists.")
    
    if not os.path.exists(parcel_with_building_path):
        fig, ax = plt.subplots(figsize=(6,6))
        gpd.GeoDataFrame([parcel], crs=parcels.crs).plot(ax=ax, facecolor="none", edgecolor="blue", linewidth=2)
        building.plot(ax=ax, facecolor="orange", edgecolor="red", alpha=0.7)
        plt.axis("equal")
        plt.savefig(parcel_with_building_path, dpi=300)
        plt.close()
   # else:
       # print(f"⏩ Skipping {parcel_with_building_path}, already exists.")

print("✅ Export complete! Images saved in /output/parcels_only and /output/parcels_with_buildings")

if not os.path.exists("output/parcel_with_building_3D.dxf"):
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    print(type(parcel))
    single_building_parcels = building_counts[building_counts == 1].index

    parcel_idx = single_building_parcels[8]  

    parcel = parcels.loc[[parcel_idx]]
    building = joined[joined.index_right == parcel_idx]


    parcel_polygon = parcel.geometry.iloc[0]
    building_polygon = building.geometry.iloc[0]

    print(parcel_polygon) 
    print(list(parcel_polygon.exterior.coords))  
    print(building_polygon)
    print(list(building_polygon.exterior.coords))


    def add_polygon(msp, poly, z=0, layer="Default"):
        points = []
        for coord in poly.exterior.coords:
            x, y = coord[:2]
            points.append((x, y, z))
        msp.add_lwpolyline(points, close=True, dxfattribs={"layer": layer})



    add_polygon(msp, parcel_polygon, z=0, layer="Parcel")

    add_polygon(msp, building_polygon, z=0.1, layer="Building")

    doc.saveas("output/parcel_with_building_3D.dxf")
    print("✅ Exported 3D-stacked DXF")

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


def pad_coords(coords, max_points):
    padded = np.full((max_points, 2), -1.0)
    padded[:len(coords), :] = coords
    return padded

def remove_padding(coords, pad_val=-1.0):
    if coords.ndim == 3:  # batch of sequences
        result = []
        for seq in coords:
            mask = (seq != pad_val).all(axis=1)
            result.append(seq[mask])
        return result
    else:  # single sequence
        mask = (coords != pad_val).all(axis=1)
        return coords[mask]


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

    parcel_padded = pad_coords(parcel_norm, MAX_POINTS)
    building_padded = pad_coords(building_norm, MAX_POINTS)
    if idx == 9:
        print( len(parcel_padded), len(building_padded))
        print("Parcel padded:", parcel_padded[:30])
        print("Building padded:", building_padded[:30])

    X_norm.append(parcel_padded)
    Y_norm.append(building_padded)
    bounds.append(pbounds)


class ParcelFootprintDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

dataset = ParcelFootprintDataset(X_norm, Y_norm)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

print("Train loader created with", len(train_loader), "batches.")
print( len(train_loader.dataset), "total samples.")
print(len(single_building_parcels))

val_ratio = 0.2
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

class CoordMapper(nn.Module):
    def __init__(self, n_points=380, n_features=2):
        super(CoordMapper, self).__init__()
        input_dim = n_points * n_features
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 1024),  nn.BatchNorm1d(1024))
        self.fc2 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
        self.fc4 = nn.Linear(256, input_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x.view(x.size(0), -1, 2)

def masked_mse(outputs, targets, pad_val=-1.0):
    mask = (targets != pad_val).float().prod(dim=2)
    loss = ((outputs - targets)**2).sum(dim=2)
    return (loss * mask).sum() / mask.sum()

def edge_loss(pred, true):
    pred_edges = pred[:, 1:] - pred[:, :-1]
    true_edges = true[:, 1:] - true[:, :-1]
    return F.mse_loss(pred_edges, true_edges)

def polygon_area(poly):
    x = poly[..., 0]
    y = poly[..., 1]
    area = 0.5 * torch.abs(torch.sum(x[:, :-1] * y[:, 1:] - x[:, 1:] * y[:, :-1], dim=1))
    return area

def area_loss(pred, true):
    pred_area = polygon_area(pred)
    true_area = polygon_area(true)
    return F.mse_loss(pred_area, true_area)


def train_model(dataset, save_path="parcel2building.pth",
                batch_size=16, lr=0.001, num_epochs=0, patience=10,
                resume=True):

    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = CoordMapper(n_points=dataset.X.shape[1], n_features=dataset.X.shape[2])

    if resume and os.path.exists(save_path):
        print(f"Resuming training from {save_path}")
        model.load_state_dict(torch.load(save_path))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)

            loss_points = masked_mse(outputs, targets)
            loss_edges  = edge_loss(outputs, targets)
            loss_areas  = area_loss(outputs, targets)

            loss = loss_points + 0.1 * loss_edges + 0.01 * loss_areas

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss_points = masked_mse(outputs, targets)
                loss_edges  = edge_loss(outputs, targets)
                loss_areas  = area_loss(outputs, targets)
                loss = loss_points + 0.1 * loss_edges + 0.01 * loss_areas
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

        scheduler.step()

    print("Training complete. Best model saved to:", save_path)
    return model

model = train_model(dataset, save_path="parcel2building.pth")

def load_model(model_path, n_points=380, n_features=2):
    model = CoordMapper(n_points=n_points, n_features=n_features)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_one(model, dataset, idx=0, plot=True):
    x, y_true = dataset[idx] 
    x = x.unsqueeze(0)  

    with torch.no_grad():
        y_pred = model(x).squeeze(0) 

    y_true = remove_padding(y_true.numpy(), pad_val=-1.0)
    x = remove_padding(x.numpy(), pad_val=-1.0)
    y_pred = remove_padding(y_pred.numpy(), pad_val=-1.0)
    print("Parcel :", x[:30])
    print("Building :", y_true[:30])
    print("Building predicted:", y_pred[:30])

    x=x[0]

    if plot:
        plt.figure(figsize=(6,6))
        plt.plot(x[:,0], x[:,1], "b.-", label="Parcel (input)")
        plt.plot(y_true[:,0], y_true[:,1], "g.-", label="Building (true)")
        plt.plot(y_pred[:,0], y_pred[:,1], "r.-", label="Building (pred)")
        plt.legend()
        plt.title(f"Sample {idx}")
        plt.show()

    return y_pred, y_true

model = load_model("parcel2building.pth", n_points=dataset.X.shape[1], n_features=dataset.X.shape[2])

y_pred, y_true = predict_one(model, dataset, idx=5)



