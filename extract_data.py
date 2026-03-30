import shutil
from pathlib import Path

def extract_data():
    # Define source paths
    # Calculate workspace root relative to this script file
    workspace_root = Path(__file__).resolve().parent.parent
    
    # Source directories
    # Please update these paths to point to your downloaded dataset folders
    # rgb_source should point to the folder containing student folders (e.g. "download/rgb_only")
    # test_source should point to the folder containing test gesture folders (e.g. "download/COMP0248_Test_data_23")
    rgb_source = workspace_root / "download" / "rgbd"
    test_source = workspace_root / "download" / "COMP0248_Test_data_23"
    
    # Destination directories
    data_dir = workspace_root / "data"
    train_dest = data_dir / "training"
    test_dest = data_dir / "test"
    
    # Check if sources exist
    if not rgb_source.exists():
        print(f"Error: Source directory {rgb_source} does not exist.")
        return
    
    if not test_source.exists():
        print(f"Error: Source directory {test_source} does not exist.")
        return

    # Clean destination directories if they exist
    if train_dest.exists():
        print(f"Cleaning existing training directory: {train_dest}")
        shutil.rmtree(train_dest)
    if test_dest.exists():
        print(f"Cleaning existing test directory: {test_dest}")
        shutil.rmtree(test_dest)

    # Create destination directories
    train_dest.mkdir(parents=True, exist_ok=True)
    test_dest.mkdir(parents=True, exist_ok=True)

    
    # Iterate over student folders
    for student_dir in rgb_source.iterdir():
        if not student_dir.is_dir():
            continue
            
        student_name = student_dir.name
        if student_name.startswith('.'):
            continue

        print(f"Processing student: {student_name}")
        
        # Iterate over gesture folders (G01_call, etc.)
        for gesture_dir in student_dir.iterdir():
            if not gesture_dir.is_dir():
                continue
                
            gesture_name = gesture_dir.name
            
            # Create destination gesture structure
            dest_gesture_dir = train_dest / gesture_name
            dest_rgb_dir = dest_gesture_dir / "rgb"
            dest_depth_dir = dest_gesture_dir / "depth"
            dest_annot_dir = dest_gesture_dir / "annotation"
            
            dest_rgb_dir.mkdir(parents=True, exist_ok=True)
            dest_depth_dir.mkdir(parents=True, exist_ok=True)
            dest_annot_dir.mkdir(parents=True, exist_ok=True)
            
            # Iterate over clip folders (clip01, etc.)
            for clip_dir in gesture_dir.iterdir():
                if not clip_dir.is_dir():
                    continue
                    
                clip_name = clip_dir.name
                
                src_rgb_dir = clip_dir / "rgb"
                src_depth_dir = clip_dir / "depth"
                src_annot_dir = clip_dir / "annotation"
                
                # Only process if all directories exist
                # Note: src_depth_dir check is added. If some students don't have depth, this might skip them.
                # Assuming all valid data has RGB, Depth, and Annotation.
                if src_rgb_dir.exists() and src_annot_dir.exists() and src_depth_dir.exists():
                    # Iterate over annotation files
                    for annot_file in src_annot_dir.glob("*.png"):
                        # Check for corresponding RGB and Depth files (case-insensitive check)
                        rgb_file = src_rgb_dir / annot_file.name
                        depth_file = src_depth_dir / annot_file.name
                        
                        found_rgb = None
                        found_depth = None
                        
                        # Find RGB
                        if rgb_file.exists():
                            found_rgb = rgb_file
                        else:
                            for f in src_rgb_dir.iterdir():
                                if f.name.lower() == annot_file.name.lower():
                                    found_rgb = f
                                    break
                        
                        # Find Depth
                        if depth_file.exists():
                            found_depth = depth_file
                        else:
                            for f in src_depth_dir.iterdir():
                                if f.name.lower() == annot_file.name.lower():
                                    found_depth = f
                                    break
                        
                        if found_rgb and found_depth:
                            # Copy all files
                            new_filename = f"{student_name}_{clip_name}_{annot_file.name}"
                            shutil.copy2(annot_file, dest_annot_dir / new_filename)
                            shutil.copy2(found_rgb, dest_rgb_dir / new_filename)
                            shutil.copy2(found_depth, dest_depth_dir / new_filename)
                        else:
                            print(f"Warning: Missing RGB or Depth file for annotation {annot_file} in {student_name}/{gesture_name}/{clip_name}")

    print("Training data extraction complete.")

    # --- Test Data Extraction ---
    print(f"Extracting test data from {test_source} to {test_dest}...")
    
    # Iterate over gesture folders directly in test_source
    for gesture_dir in test_source.iterdir():
        if not gesture_dir.is_dir():
            continue
            
        gesture_name = gesture_dir.name
        
        # Create destination gesture structure
        dest_gesture_dir = test_dest / gesture_name
        dest_rgb_dir = dest_gesture_dir / "rgb"
        dest_depth_dir = dest_gesture_dir / "depth"
        dest_annot_dir = dest_gesture_dir / "annotation"
        
        dest_rgb_dir.mkdir(parents=True, exist_ok=True)
        dest_depth_dir.mkdir(parents=True, exist_ok=True)
        dest_annot_dir.mkdir(parents=True, exist_ok=True)
        
        # Iterate over clip folders (clip01, etc.)
        for clip_dir in gesture_dir.iterdir():
            if not clip_dir.is_dir():
                continue
                
            clip_name = clip_dir.name
            
            src_rgb_dir = clip_dir / "rgb"
            src_depth_dir = clip_dir / "depth"
            src_annot_dir = clip_dir / "annotation"
            
            # Only process if all directories exist
            if src_rgb_dir.exists() and src_annot_dir.exists() and src_depth_dir.exists():
                # Iterate over annotation files
                for annot_file in src_annot_dir.glob("*.png"):
                    # Check for corresponding RGB and Depth files
                    rgb_file = src_rgb_dir / annot_file.name
                    depth_file = src_depth_dir / annot_file.name
                    
                    found_rgb = None
                    found_depth = None
                    
                    if rgb_file.exists():
                        found_rgb = rgb_file
                    else:
                        for f in src_rgb_dir.iterdir():
                            if f.name.lower() == annot_file.name.lower():
                                found_rgb = f
                                break
                                
                    if depth_file.exists():
                        found_depth = depth_file
                    else:
                        for f in src_depth_dir.iterdir():
                            if f.name.lower() == annot_file.name.lower():
                                found_depth = f
                                break
                    
                    if found_rgb and found_depth:
                        # Copy all files
                        new_filename = f"Test_{clip_name}_{annot_file.name}"
                        shutil.copy2(annot_file, dest_annot_dir / new_filename)
                        shutil.copy2(found_rgb, dest_rgb_dir / new_filename)
                        shutil.copy2(found_depth, dest_depth_dir / new_filename)
                    else:
                        print(f"Warning: Missing RGB or Depth file for annotation {annot_file} in Test/{gesture_name}/{clip_name}")

    print("Test data extraction complete.")

if __name__ == "__main__":
    extract_data()
