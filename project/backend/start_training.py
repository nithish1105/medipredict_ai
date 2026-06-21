"""
Quick training launcher with diagnostics
==========================================
Checks prerequisites and launches training with optimal settings.
"""

import os
import sys

def check_prerequisites():
    """Check if all required packages and datasets are available."""
    print("=" * 70)
    print("  TRAINING PREREQUISITES CHECK")
    print("=" * 70)
    
    # Check Python version
    print(f"\n✓ Python version: {sys.version.split()[0]}")
    
    # Check packages
    packages = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "numpy": "NumPy",
        "sklearn": "scikit-learn",
        "PIL": "Pillow"
    }
    
    missing = []
    for pkg, name in packages.items():
        try:
            __import__(pkg)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - MISSING")
            missing.append(pkg)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install torch torchvision numpy scikit-learn pillow")
        return False
    
    # Check datasets
    print("\nDataset availability:")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(base_dir), "datasets")
    
    datasets = [
        "archive (1).zip",
        "archive (2).zip", 
        "archive (3).zip",
        "archive (4).zip"
    ]
    
    found_datasets = []
    for ds in datasets:
        path = os.path.join(data_dir, ds)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✓ {ds:20s} ({size_mb:.1f} MB)")
            found_datasets.append(ds)
        else:
            print(f"✗ {ds:20s} - NOT FOUND")
    
    if len(found_datasets) < 2:
        print(f"\n⚠️  Only {len(found_datasets)} dataset(s) found.")
        print("   Training will proceed but accuracy may be lower.")
        print(f"   Datasets should be in: {data_dir}")
    
    # Check disk space
    try:
        import shutil
        stat = shutil.disk_usage(base_dir)
        free_gb = stat.free / (1024 ** 3)
        print(f"\n✓ Free disk space: {free_gb:.1f} GB")
        
        if free_gb < 5:
            print("  ⚠️  Warning: Less than 5 GB free - feature cache may fail")
    except:
        pass
    
    print("\n" + "=" * 70)
    
    if missing:
        return False
    
    return True


def run_training():
    """Launch the training script."""
    print("\nStarting training...\n")
    
    # Import and run
    import train_accurate
    train_accurate.main()


if __name__ == "__main__":
    if check_prerequisites():
        print("\n✅ All prerequisites met!")
        
        response = input("\nStart training? This may take 40-80 minutes. [y/N]: ")
        if response.lower() in ['y', 'yes']:
            run_training()
        else:
            print("\nTraining cancelled.")
            print("\nTo train later, run:")
            print("  python train_accurate.py")
    else:
        print("\n❌ Prerequisites not met. Please install missing packages.")
        sys.exit(1)
