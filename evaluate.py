import torch
import numpy as np
import os
import time
from dataloader import get_dataloader
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import models
import sys
import importlib

class Config:
    _src_dir = os.path.dirname(os.path.abspath(__file__))
    _base_dir = os.path.dirname(_src_dir)
    
    data_dir = os.path.join(_base_dir, 'data')
    results_dir = os.path.join(_base_dir, 'results')
    
    # Paths to saved weights
    pointnet_path = os.path.join(_src_dir, 'weights', 'pointnet')
    dgcnn_path = os.path.join(_src_dir, 'weights', 'DGCNN', 'best_dgcnn.pth')
    
    batch_size = 24
    num_points = 2048

def plot_confusion_matrix(y_true, y_pred, classes, save_path, title):
    """
    Plot and save the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model(model, loader, device, model_name, results_dir):
    """
    Evaluate a specific model and calculate metrics including efficiency.
    """
    print(f"\n{'='*40}")
    print(f"Evaluating {model_name}...")
    print(f"{'='*40}")
    
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # Variables for efficiency measurement
    total_inference_time = 0.0
    num_samples = 0
    
    # Warm up GPU (important for accurate timing)
    dummy_input = torch.randn(1, 3, Config.num_points).to(device)
    with torch.no_grad():
        for _ in range(10):
            if model_name == "PointNet":
                _ = model(dummy_input)
            else:
                _ = model(dummy_input)
    
    with torch.no_grad():
        for points, labels in loader:
            points = points.to(device)
            labels = labels.to(device)
            
            # Start timing
            start_time = time.time()
            
            # Forward pass
            if model_name == "PointNet":
                cls_logits, _ = model(points)
            else:
                cls_logits = model(points)
                
            # End timing
            end_time = time.time()
            total_inference_time += (end_time - start_time)
            num_samples += points.size(0)
            
            # Calculate predictions
            _, preds = torch.max(cls_logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate Metrics
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # Calculate Efficiency
    avg_inference_time_ms = (total_inference_time / num_samples) * 1000
    fps = 1.0 / (total_inference_time / num_samples)
    num_params = count_parameters(model)
    
    print(f"Performance Metrics:")
    print(f"  - Overall Accuracy: {acc*100:.2f}%")
    print(f"  - Macro F1 Score:   {macro_f1:.4f}")
    print(f"\nEfficiency Metrics:")
    print(f"  - Parameters:       {num_params / 1e6:.2f} M")
    print(f"  - Inference Time:   {avg_inference_time_ms:.2f} ms / sample")
    print(f"  - FPS:              {fps:.2f} frames / second")
    
    # Save Confusion Matrix
    os.makedirs(results_dir, exist_ok=True)
    dataset = loader.dataset
    if hasattr(dataset, 'dataset'):
        classes = dataset.dataset.classes
    else:
        classes = dataset.classes
        
    cm_path = os.path.join(results_dir, f'cm_{model_name}.png')
    plot_confusion_matrix(all_labels, all_preds, classes, cm_path, title=f'{model_name} Confusion Matrix')
    
    return {
        'acc': acc,
        'f1': macro_f1,
        'params_M': num_params / 1e6,
        'fps': fps
    }

def main():
    args = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Test Data
    print(f"Loading test data from {args.data_dir}...")
    test_loader = get_dataloader(
        args.data_dir, 
        split='test', 
        batch_size=args.batch_size, 
        shuffle=False,
        num_points=args.num_points
    )
    print(f"Test batches: {len(test_loader)}")
    
    results = {}
    
    # --- Evaluate PointNet (Baseline) ---
    print("\nLoading PointNet...")
    
    # We need to import the model dynamically like in train_pointnet.py
    sys.path.append(os.path.join(args._src_dir, 'models'))
    pointnet_module = importlib.import_module('pointnet_cls')
    
    # Note: We need to find the latest timestamp folder for PointNet
    pointnet_base_dir = args.pointnet_path
    if os.path.exists(pointnet_base_dir):
        # Get the most recent folder
        subfolders = [f.path for f in os.scandir(pointnet_base_dir) if f.is_dir()]
        if subfolders:
            latest_folder = max(subfolders, key=os.path.getmtime)
            pointnet_weight_path = os.path.join(latest_folder, 'checkpoints', 'best_model.pth')
            
            if os.path.exists(pointnet_weight_path):
                # Initialize using the original get_model function
                model_pn = pointnet_module.get_model(k=10, normal_channel=False).to(device)
                # Load the state dict correctly from the checkpoint dictionary
                checkpoint = torch.load(pointnet_weight_path, map_location=device, weights_only=False)
                model_pn.load_state_dict(checkpoint['model_state_dict'])
                results['PointNet'] = evaluate_model(model_pn, test_loader, device, "PointNet", args.results_dir)
            else:
                print(f"PointNet weights not found at {pointnet_weight_path}")
        else:
            print("No PointNet training folders found.")
    else:
        print("PointNet weights directory not found.")
        
        # --- Evaluate DGCNN (Advanced Model) ---
    print("\nLoading DGCNN...")
    from model_DGCNN import DGCNN
    if os.path.exists(args.dgcnn_path):
        model_dgcnn = DGCNN(k=20, output_channels=10).to(device)
        model_dgcnn.load_state_dict(torch.load(args.dgcnn_path, map_location=device))
        results['DGCNN'] = evaluate_model(model_dgcnn, test_loader, device, "DGCNN", args.results_dir)
    else:
        print(f"DGCNN weights not found at {args.dgcnn_path}")

if __name__ == '__main__':
    main()