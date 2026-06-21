# Install PyTorch and required dependencies for training
# Run this script before training the model

Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host "  Installing PyTorch and Training Dependencies" -ForegroundColor Yellow
Write-Host ("=" * 70) -ForegroundColor Cyan

Write-Host "`nChecking current environment..." -ForegroundColor Green

# Check Python version
$pythonVersion = python --version 2>&1
Write-Host "Python: $pythonVersion" -ForegroundColor White

# Install PyTorch (CPU version - faster download, works on all systems)
Write-Host "`nInstalling PyTorch (CPU version)..." -ForegroundColor Green
Write-Host "This may take 2-5 minutes depending on your internet speed.`n" -ForegroundColor Yellow

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ PyTorch installed successfully!" -ForegroundColor Green
} else {
    Write-Host "`n✗ PyTorch installation failed!" -ForegroundColor Red
    Write-Host "Try running manually:" -ForegroundColor Yellow
    Write-Host "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu" -ForegroundColor White
    exit 1
}

# Install other dependencies (if not already present)
Write-Host "`nInstalling additional dependencies..." -ForegroundColor Green
pip install -r requirements_training.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ All dependencies installed!" -ForegroundColor Green
} else {
    Write-Host "`n! Some dependencies may already be installed (this is OK)" -ForegroundColor Yellow
}

# Verify installation
Write-Host "`nVerifying installation..." -ForegroundColor Green
python -c "import torch; import torchvision; import numpy; import sklearn; from PIL import Image; print('✓ All packages working correctly!')"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n" -NoNewline
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  Installation Complete!" -ForegroundColor Green
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "`nYou're ready to train the model!" -ForegroundColor Yellow
    Write-Host "`nNext steps:" -ForegroundColor White
    Write-Host "  1. Ensure datasets are in ../datasets/ folder" -ForegroundColor Gray
    Write-Host "  2. Run: python start_training.py" -ForegroundColor Gray
    Write-Host "     OR: python train_accurate.py" -ForegroundColor Gray
    Write-Host "`nEstimated training time: 45-75 minutes (first run)" -ForegroundColor Yellow
    Write-Host "Target accuracy: 95%+ (up from ~85% previously)`n" -ForegroundColor Green
} else {
    Write-Host "`n✗ Installation verification failed!" -ForegroundColor Red
    Write-Host "Please check error messages above." -ForegroundColor Yellow
    exit 1
}
