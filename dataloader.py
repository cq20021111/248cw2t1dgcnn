import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class HandGestureDataset(Dataset):
    """
    Custom hand gesture dataset class for CW2.
    Loads ONLY depth images and converts them to 3D point clouds.
    10-Class classification only (no segmentation & bounding box).
    """
    def __init__(self, root_dir, split='training', num_points=1024, augment=None):
        """
        Initialize the dataset.
        Args:
            root_dir (string): Root directory of the dataset.
            split (string): 'training' or 'test'.
            num_points (int): Number of points to sample for each point cloud.
            augment (bool, optional): Whether to apply 3D data augmentation. Defaults to True if split='training'.
        """
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        
        if augment is None:
            self.augment = (split == 'training')
        else:
            self.augment = augment

        # Define 10 hand gesture class names
        self.classes = [
            'G01_call', 'G02_dislike', 'G03_like', 'G04_ok', 'G05_one',
            'G06_palm', 'G07_peace', 'G08_rock', 'G09_stop', 'G10_three'
        ]
        # Create class name -> index mapping dictionary
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []

        # Iterate through directory structure to load data
        data_path = os.path.join(root_dir, split)
        if not os.path.exists(data_path):
            raise ValueError(f"Directory {data_path} does not exist.")

        for cls_name in self.classes:
            cls_dir = os.path.join(data_path, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"Warning: Class directory {cls_dir} does not exist.")
                continue
            
            depth_dir = os.path.join(cls_dir, 'depth')
            if not os.path.exists(depth_dir):
                print(f"Warning: Depth directory {depth_dir} does not exist.")
                continue

            # Load only depth images
            for img_name in os.listdir(depth_dir):
                if img_name.endswith('.png'):
                    depth_path = os.path.join(depth_dir, img_name)
                    self.samples.append({
                        'depth_path': depth_path,
                        'label': self.class_to_idx[cls_name]
                    })

    def depth_to_point_cloud(self, depth_np):
        """
        Convert a 2D depth image to a 3D point cloud.
        """
        h, w = depth_np.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        u = u.flatten()
        v = v.flatten()
        z = depth_np.flatten()
        
        # Filter out background. Assuming depth > 0 is foreground.
        valid_idx = z > 0
        if not np.any(valid_idx):
            # Fallback if all depth values are 0
            valid_idx = np.ones_like(z, dtype=bool)
            
        u = u[valid_idx]
        v = v[valid_idx]
        z = z[valid_idx]
        
        # Normalize u, v to [-1, 1] range based on image dimensions
        x = (u - w / 2) / (w / 2)
        y = (v - h / 2) / (h / 2)
        
        # Convert z to float and normalize to [-1, 1] range roughly
        # Assuming depth values are typically in 0-255 range for 8-bit images
        z = z.astype(np.float32)
        if z.max() > 0:
            z = (z - z.min()) / (z.max() - z.min()) * 2 - 1
        
        points = np.stack((x, y, z), axis=-1) # Shape: (N, 3)
        
        # Normalize point cloud: center at origin and scale to unit sphere
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_distance = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if max_distance > 0:
            points = points / max_distance
            
        # Sample a fixed number of points (e.g., 1024)
        if len(points) >= self.num_points:
            choice = np.random.choice(len(points), self.num_points, replace=False)
        else:
            # Pad with replacement if not enough points
            choice = np.random.choice(len(points), self.num_points, replace=True)
            
        points = points[choice, :]
        return points

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get sample at the specified index.
        Returns:
            points_tensor: 3D Point Cloud tensor (num_points, 3)
            label: Class label index
        """
        sample_info = self.samples[idx]
        depth_path = sample_info['depth_path']
        label = sample_info['label']

        # Load depth image (usually single channel L or I)
        depth = Image.open(depth_path)
        depth_np = np.array(depth)
        
        # Convert depth to point cloud
        points = self.depth_to_point_cloud(depth_np)
        
        # Apply 3D Data Augmentation (training only; helps reduce overfitting)
        if self.augment:
            # Random rotation around Z-axis (in-plane tilt of the hand in x-y)
            theta = np.random.uniform(-np.pi / 16, np.pi / 16)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            rot_matrix = np.array(
                [
                    [cos_t, -sin_t, 0],
                    [sin_t, cos_t, 0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            points = np.dot(points, rot_matrix)

            # Random jitter (small Gaussian noise on coordinates)
            jitter = np.random.normal(0, 0.01, size=points.shape).astype(np.float32)
            points = np.clip(points + jitter, -1.0, 1.0)
        
        # Convert to Tensor and transpose to (Channels, Num_Points) for PointNet
        # Original shape: (num_points, 3) -> New shape: (3, num_points)
        points_tensor = torch.from_numpy(points).float().transpose(0, 1)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return points_tensor, label_tensor

def get_dataloader(root_dir, split='training', batch_size=32, shuffle=True, num_workers=6, val_split=0.0, num_points=1024):
    """
    Create and return DataLoader instance.
    If val_split > 0.0, split dataset into train and validation sets, and return (train_loader, val_loader).
    Otherwise return a single dataloader.
    """
    if val_split > 0.0:
        # Get dataset size
        full_dataset = HandGestureDataset(root_dir, split=split, num_points=num_points)
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * val_split)
        train_size = dataset_size - val_size
        
        # Use Generator to fix seed, ensuring consistent split every time
        generator = torch.Generator().manual_seed(42)
        
        # Generate random indices
        indices = torch.randperm(dataset_size, generator=generator).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create two independent dataset objects
        # Train set: enable augmentation
        train_dataset = HandGestureDataset(root_dir, split=split, num_points=num_points, augment=True)
        # Validation set: disable augmentation
        val_dataset = HandGestureDataset(root_dir, split=split, num_points=num_points, augment=False)
        
        # Use Subset to specify indices
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        val_subset = torch.utils.data.Subset(val_dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader
    else:
        # If not splitting validation set
        dataset = HandGestureDataset(root_dir, split=split, num_points=num_points)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return dataloader
