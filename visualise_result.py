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

def visualize_point_cloud(ax, points, title, true_label, pred_label, is_correct, pred_prob):
    """
    Visualize a single 3D point cloud on a given matplotlib axis.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Scatter plot
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=2, alpha=0.8)
    
    # Set labels and title
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    
    # Color title based on correctness
    title_color = 'green' if is_correct else 'red'
    ax.set_title(
        f"{title}\nTrue: {true_label}\nPred: {pred_label}\nProb: {pred_prob*100:.1f}%",
        color=title_color,
        fontsize=8,
        linespacing=1.15,
        pad=4,
    )
    
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
    
    # Get two batches of data to increase candidate samples
    data_iter = iter(test_loader)
    batch_points = []
    batch_labels = []
    for _ in range(2):
        try:
            p, l = next(data_iter)
            batch_points.append(p)
            batch_labels.append(l)
        except StopIteration:
            break

    if not batch_points:
        print("No test samples available for visualization.")
        return

    # Keep full candidate pool on CPU for visualization and indexing
    points_batch = torch.cat(batch_points, dim=0)
    labels_batch = torch.cat(batch_labels, dim=0)
    
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
            
            # Run inference batch-by-batch to avoid large peak GPU memory
            preds_list = []
            pred_probs_list = []
            for p, _ in zip(batch_points, batch_labels):
                p = p.to(device)
                if model_name == "PointNet":
                    cls_logits, _ = model(p)
                else:
                    cls_logits = model(p)

                probs = torch.softmax(cls_logits, dim=1)
                pred_prob, pred = torch.max(probs, 1)
                preds_list.append(pred.cpu())
                pred_probs_list.append(pred_prob.cpu())

            preds = torch.cat(preds_list, dim=0)
            pred_probs = torch.cat(pred_probs_list, dim=0)
            
            # Find correct and incorrect indices
            correct_idx = (preds == labels_batch).nonzero(as_tuple=True)[0]
            incorrect_idx = (preds != labels_batch).nonzero(as_tuple=True)[0]

            # Best: top-3 correctly predicted samples with highest prediction probability
            best_idx = []
            if len(correct_idx) > 0:
                correct_conf = pred_probs[correct_idx]
                best_order = torch.argsort(correct_conf, descending=True)
                best_idx = correct_idx[best_order[: min(3, len(correct_idx))]].cpu().tolist()

            # Worst: top-3 incorrectly predicted samples with highest (wrong) prediction probability
            worst_idx = []
            if len(incorrect_idx) > 0:
                incorrect_conf = pred_probs[incorrect_idx]
                worst_order = torch.argsort(incorrect_conf, descending=True)
                worst_idx = incorrect_idx[worst_order[: min(3, len(incorrect_idx))]].cpu().tolist()

            # Fallback to keep 6 plots whenever possible
            if len(best_idx) < 3:
                remaining_pool = [i for i in correct_idx.cpu().tolist() if i not in best_idx]
                best_idx.extend(remaining_pool[: 3 - len(best_idx)])
            if len(worst_idx) < 3:
                remaining_pool = [i for i in incorrect_idx.cpu().tolist() if i not in worst_idx]
                worst_idx.extend(remaining_pool[: 3 - len(worst_idx)])

            selected_idx = best_idx + worst_idx
            if not selected_idx:
                print("No samples to visualize.")
                continue

            # Plotting: 2 rows x 3 columns (Best 3 on top, Worst 3 on bottom)
            fig = plt.figure(figsize=(11, 7.6))
            fig.suptitle(
                f'{model_name} Predictions (Top: Best 3 Correct, Bottom: Worst 3 Incorrect)',
                fontsize=13,
                y=0.97
            )

            for i, idx in enumerate(selected_idx[:6]):
                ax = fig.add_subplot(2, 3, i + 1, projection='3d')
                
                # Extract point cloud and convert back to (Num_Points, 3) for plotting
                pc = points_batch[idx].transpose(0, 1).numpy()
                true_label = class_names[labels_batch[idx].item()]
                pred_label = class_names[preds[idx].item()]
                is_correct = (labels_batch[idx] == preds[idx]).item()
                prob = pred_probs[idx].item()
                
                if i < 3:
                    title = f"Best {i+1} (idx {idx})"
                else:
                    title = f"Worst {i-2} (idx {idx})"
                visualize_point_cloud(ax, pc, title, true_label, pred_label, is_correct, prob)

            plt.subplots_adjust(left=0, right=1, top=0.85, bottom=0, wspace=0, hspace=0.1)
            save_path = os.path.join(args.output_dir, f'visualisation_{model_name}.png')
            plt.savefig(save_path, dpi=150)
            print(f"Visualization saved to {save_path}")
            plt.close()

if __name__ == '__main__':
    visualize_predictions()