import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataloader import get_dataloader

def visualize_point_cloud(points, title="Point Cloud"):
    """
    Visualize a single 3D point cloud using matplotlib.
    Args:
        points (numpy.ndarray): Shape (N, 3)
        title (str): Title of the plot
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract X, Y, Z coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Scatter plot
    # Using z for color mapping to make depth more visible
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=3, alpha=0.8)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set consistent axis limits for better comparison
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='Depth (Z)')
    
    plt.show()

def main():
    # Define data directory relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data')
    
    print(f"Loading data from: {data_dir}")
    
    # Get dataloader (we only need the train_loader for visualization)
    # Set shuffle=True to see different examples each time
    train_loader, _ = get_dataloader(
        root_dir=data_dir, 
        split='training', 
        batch_size=6, 
        val_split=0.1, # Just to get the split working
        num_points=4096
    )
    
    # Define class names for labeling
    classes = [
        'G01_call', 'G02_dislike', 'G03_like', 'G04_ok', 'G05_one',
        'G06_palm', 'G07_peace', 'G08_rock', 'G09_stop', 'G10_three'
    ]
    
    # Get a single batch
    data_iter = iter(train_loader)
    points_batch, labels_batch = next(data_iter)
    
    print(f"Batch points shape: {points_batch.shape}") # Expected: (Batch, Channels, Num_Points) -> (4, 3, 1024)
    print(f"Batch labels shape: {labels_batch.shape}")
    
    # Visualize the first few point clouds in the batch
    num_to_visualize = min(4, points_batch.shape[0])
    
    for i in range(num_to_visualize):
        # Extract single point cloud
        points_tensor = points_batch[i]
        label_idx = labels_batch[i].item()
        class_name = classes[label_idx]
        
        # Convert from (Channels, Num_Points) to (Num_Points, Channels) for plotting
        # and convert to numpy array
        points_np = points_tensor.transpose(0, 1).numpy()
        
        print(f"\nVisualizing sample {i+1}: Class '{class_name}' (Label {label_idx})")
        print(f"Points shape: {points_np.shape}, Min: {points_np.min():.3f}, Max: {points_np.max():.3f}")
        
        visualize_point_cloud(points_np, title=f"Gesture: {class_name}")

if __name__ == "__main__":
    main()
