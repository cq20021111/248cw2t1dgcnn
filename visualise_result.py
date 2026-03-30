import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from dataloader import get_dataloader

# Import models
from model_DGCNN import DGCNN

class Config:
    _src_dir = os.path.dirname(os.path.abspath(__file__))
    _base_dir = os.path.dirname(_src_dir)
    
    data_dir = os.path.join(_base_dir, 'data')
    output_dir = os.path.join(_base_dir, 'results')
    
    # Paths to saved weights
    pointnet_path = os.path.join(_src_dir, 'weights', 'pointnet')
    dgcnn_path = os.path.join(_src_dir, 'weights', 'DGCNN', 'best_dgcnn.pth')
    
    batch_size = 24
    num_points = 4096

def visualize_point_cloud(ax, points, title, true_label, pred_label, is_correct):
    """
    Visualize a single 3D point cloud on a given matplotlib axis.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Scatter plot
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=5, alpha=0.8)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Color title based on correctness
    title_color = 'green' if is_correct else 'red'
    ax.set_title(f"{title}\nTrue: {true_label}\nPred: {pred_label}", color=title_color, fontsize=10)
    
    # Set consistent axis limits (zoomed in by reducing the limits from 1.2 to 0.6)
    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([-0.6, 0.6])
    ax.set_zlim([-0.6, 0.6])
    
    # Rotate the view
    # elev=90 looks down from the Z-axis (effectively rotating 90 deg around X from default)
    # azim=90 rotates 180 degrees around Z compared to the previous azim=-90
    ax.view_init(elev=90, azim=90)
    
    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

def get_latest_pointnet_weights(base_dir):
    if os.path.exists(base_dir):
        subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
        if subfolders:
            latest_folder = max(subfolders, key=os.path.getmtime)
            return os.path.join(latest_folder, 'checkpoints', 'best_model.pth')
    return None

def visualize_predictions():
    """
    Visualize model prediction results for Point Cloud Classification.
    Displays a few correct and incorrect predictions for both models.
    """
    args = Config()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load data
    print("Loading test data...")
    test_loader = get_dataloader(args.data_dir, split='test', batch_size=args.batch_size, shuffle=True, num_points=args.num_points)
    
    # Get a single batch of data
    points_batch, labels_batch = next(iter(test_loader))
    points_batch = points_batch.to(device)
    labels_batch = labels_batch.to(device)
    
    class_names = [
        'call', 'dislike', 'like', 'ok', 'one',
        'palm', 'peace', 'rock', 'stop', 'three'
    ]
    
    # 2. Load Models
    pointnet_weight_path = get_latest_pointnet_weights(args.pointnet_path)
    
    models = {}
    
    if pointnet_weight_path and os.path.exists(pointnet_weight_path):
        import sys
        import importlib
        sys.path.append(os.path.join(args._src_dir, 'models'))
        pointnet_module = importlib.import_module('pointnet_cls')
        model_pn = pointnet_module.get_model(k=10, normal_channel=False).to(device)
        checkpoint = torch.load(pointnet_weight_path, map_location=device, weights_only=False)
        model_pn.load_state_dict(checkpoint['model_state_dict'])
        model_pn.eval()
        models['PointNet'] = model_pn
        print("PointNet loaded successfully.")
    else:
        print("PointNet weights not found.")
        
    if os.path.exists(args.dgcnn_path):
        model_dgcnn = DGCNN(k=20, output_channels=10).to(device)
        model_dgcnn.load_state_dict(torch.load(args.dgcnn_path, map_location=device))
        model_dgcnn.eval()
        models['DGCNN'] = model_dgcnn
        print("DGCNN loaded successfully.")
    else:
        print("DGCNN weights not found.")

    if not models:
        print("No models available for visualization.")
        return

    # 3. Get Predictions and Visualize
    with torch.no_grad():
        for model_name, model in models.items():
            print(f"\nProcessing {model_name}...")
            
            if model_name == "PointNet":
                cls_logits, _ = model(points_batch)
            else:
                cls_logits = model(points_batch)
                
            _, preds = torch.max(cls_logits, 1)
            
            # Find correct and incorrect indices
            correct_idx = (preds == labels_batch).nonzero(as_tuple=True)[0]
            incorrect_idx = (preds != labels_batch).nonzero(as_tuple=True)[0]
            
            # Select up to 2 correct and 2 incorrect samples
            selected_idx = []
            if len(correct_idx) > 0:
                selected_idx.extend(correct_idx[:min(2, len(correct_idx))].cpu().numpy())
            if len(incorrect_idx) > 0:
                selected_idx.extend(incorrect_idx[:min(2, len(incorrect_idx))].cpu().numpy())
                
            if not selected_idx:
                print("No samples to visualize.")
                continue
                
            # Plotting
            num_display = len(selected_idx)
            fig = plt.figure(figsize=(4 * num_display, 5))
            fig.suptitle(f'{model_name} Predictions (Green=Correct, Red=Incorrect)', fontsize=16)
            
            for i, idx in enumerate(selected_idx):
                ax = fig.add_subplot(1, num_display, i + 1, projection='3d')
                
                # Extract point cloud and convert back to (Num_Points, 3) for plotting
                pc = points_batch[idx].cpu().transpose(0, 1).numpy()
                true_label = class_names[labels_batch[idx].item()]
                pred_label = class_names[preds[idx].item()]
                is_correct = (labels_batch[idx] == preds[idx]).item()
                
                visualize_point_cloud(ax, pc, f"Sample {idx}", true_label, pred_label, is_correct)
                
            plt.tight_layout()
            save_path = os.path.join(args.output_dir, f'visualisation_{model_name}.png')
            plt.savefig(save_path, dpi=150)
            print(f"Visualization saved to {save_path}")
            plt.close()

if __name__ == '__main__':
    visualize_predictions()