import torch
import torch.optim as optim
import os
from dataloader import get_dataloader
from model_DGCNN import DGCNN

class Config:
    # Get the directory where the current script is located (src)
    _src_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (dataset_and_code)
    _base_dir = os.path.dirname(_src_dir)
    
    # Deduce data directory and weights save directory
    data_dir = os.path.join(_base_dir, 'data')
    save_dir = os.path.join(_src_dir, 'weights', 'DGCNN')
    
    # Training hyperparameters
    epochs = 200         # Total number of training epochs
    batch_size = 24      # Batch size (DGCNN uses more memory than PointNet, may need smaller batch size)
    lr = 0.0004           # Learning rate
    val_split = 0.15     # Validation set ratio
    num_points = 2048    # Number of points to sample per point cloud
    k = 20               # Number of nearest neighbors for DGCNN EdgeConv
    
    # Early Stopping parameters
    patience = 20        # Number of epochs to tolerate no decrease in validation loss
    min_delta = 0.0005   # Minimum threshold for validation loss decrease
    
    # Resume training flag
    resume = True       
    resume_path = os.path.join(save_dir, 'best_dgcnn.pth')

def validate(model, val_loader, device, criterion):
    """
    Evaluate DGCNN performance on the validation set.
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for points, labels in val_loader:
            points = points.to(device)
            labels = labels.to(device)
            
            # Forward pass
            preds = model(points)
            
            # Calculate loss
            loss = criterion(preds, labels)
            val_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    num_batches = len(val_loader)
    accuracy = 100 * correct / total
    return val_loss / num_batches, accuracy

def train():
    """
    Main training function for DGCNN (Advanced Model).
    """
    args = Config()
    
    if not torch.cuda.is_available():
        print("Warning: No CUDA GPU available. Using CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"Using device: {device}")
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"Loading data from {args.data_dir}...")
    train_loader, val_loader = get_dataloader(
        args.data_dir, 
        split='training', 
        batch_size=args.batch_size, 
        val_split=args.val_split,
        num_points=args.num_points
    )
    print(f"Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Initialize DGCNN Model
    model = DGCNN(k=args.k, output_channels=10).to(device)
    
    if args.resume and os.path.exists(args.resume_path):
        print(f"Resuming training from {args.resume_path}...")
        model.load_state_dict(torch.load(args.resume_path, map_location=device))
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Cosine Annealing Learning Rate Scheduler (works well with DGCNN)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    # Loss Function (CrossEntropyLoss with label smoothing to prevent overfitting)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print("Starting DGCNN training...")
    best_val_acc = 0.0
    early_stop_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (points, labels) in enumerate(train_loader):
            points = points.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            preds = model(points)
            
            # Calculate loss
            loss = criterion(preds, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], "
                      f"Train Loss: {loss.item():.4f}")
        
        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation Phase
        avg_val_loss, val_acc = validate(model, val_loader, device, criterion)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Step the scheduler
        scheduler.step()
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_dgcnn.pth'))
            print(f"New best model saved with Val Acc: {best_val_acc:.2f}%")
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                print(f"Early stopping triggered. Best Val Acc: {best_val_acc:.2f}%")
                break

if __name__ == '__main__':
    train()