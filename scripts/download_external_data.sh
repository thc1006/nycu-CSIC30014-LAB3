#!/bin/bash
# Download External Medical Imaging Datasets for 91+ Breakthrough
# CheXpert, MIMIC-CXR, and MedSAM

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/external_data"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=========================================="
echo "External Data Download Script"
echo "=========================================="
echo "Target directory: $DATA_DIR"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check disk space
check_disk_space() {
    local required_gb=$1
    local available_gb=$(df -BG "$DATA_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')

    if [ "$available_gb" -lt "$required_gb" ]; then
        echo -e "${RED}❌ Insufficient disk space!${NC}"
        echo "Required: ${required_gb}GB, Available: ${available_gb}GB"
        return 1
    fi
    echo -e "${GREEN}✓ Sufficient disk space: ${available_gb}GB available${NC}"
    return 0
}

# ==========================================
# 1. Download MedSAM (2.4GB)
# ==========================================
echo ""
echo "=========================================="
echo "[1/3] Downloading MedSAM Model"
echo "=========================================="
echo "Size: ~2.4GB"
echo "Source: https://github.com/bowang-lab/MedSAM"
echo ""

check_disk_space 5 || exit 1

if [ ! -f "medsam_vit_b.pth" ]; then
    echo "Downloading MedSAM checkpoint..."
    wget -c https://huggingface.co/wanglab/medsam/resolve/main/medsam_vit_b.pth \
        -O medsam_vit_b.pth
    echo -e "${GREEN}✓ MedSAM downloaded successfully${NC}"
else
    echo -e "${YELLOW}⚠ MedSAM already exists, skipping${NC}"
fi

# ==========================================
# 2. Download CheXpert (11GB downsampled)
# ==========================================
echo ""
echo "=========================================="
echo "[2/3] Downloading CheXpert Dataset"
echo "=========================================="
echo "Size: ~11GB (downsampled version)"
echo "Source: https://stanfordmlgroup.github.io/competitions/chexpert/"
echo ""
echo -e "${YELLOW}NOTE: CheXpert requires registration at:${NC}"
echo "https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2"
echo ""

check_disk_space 15 || exit 1

read -p "Have you obtained the download link from Stanford? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Please paste the download link: " CHEXPERT_LINK

    if [ ! -d "CheXpert-v1.0-small" ]; then
        echo "Downloading CheXpert..."
        wget -c "$CHEXPERT_LINK" -O chexpert.zip
        echo "Extracting..."
        unzip -q chexpert.zip
        rm chexpert.zip
        echo -e "${GREEN}✓ CheXpert downloaded and extracted${NC}"
    else
        echo -e "${YELLOW}⚠ CheXpert already exists, skipping${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Skipping CheXpert download${NC}"
    echo "To download later:"
    echo "1. Register at: https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2"
    echo "2. Run this script again"
fi

# ==========================================
# 3. Download MIMIC-CXR-JPG (subset)
# ==========================================
echo ""
echo "=========================================="
echo "[3/3] MIMIC-CXR-JPG Information"
echo "=========================================="
echo "Size: ~100GB (full), ~10GB (sample)"
echo "Source: https://physionet.org/content/mimic-cxr-jpg/2.0.0/"
echo ""
echo -e "${YELLOW}NOTE: MIMIC-CXR requires credentialing:${NC}"
echo "1. Complete CITI training: https://physionet.org/about/citi-course/"
echo "2. Sign data use agreement"
echo "3. Get approval (takes 1-3 days)"
echo ""

read -p "Do you have PhysioNet credentials and want to download? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Installing PhysioNet downloader..."
    pip install wfdb -q

    read -p "Enter your PhysioNet username: " PHYSIONET_USER
    read -sp "Enter your PhysioNet password: " PHYSIONET_PASS
    echo ""

    if [ ! -d "mimic-cxr-jpg" ]; then
        echo "Downloading MIMIC-CXR-JPG (this will take several hours)..."
        wget -r -N -c -np --user "$PHYSIONET_USER" --password "$PHYSIONET_PASS" \
            https://physionet.org/files/mimic-cxr-jpg/2.0.0/
        echo -e "${GREEN}✓ MIMIC-CXR-JPG download started${NC}"
    else
        echo -e "${YELLOW}⚠ MIMIC-CXR-JPG already exists, skipping${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Skipping MIMIC-CXR download${NC}"
    echo "To download later:"
    echo "1. Complete CITI training"
    echo "2. Get PhysioNet approval"
    echo "3. Run this script again"
fi

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "Download Summary"
echo "=========================================="
echo ""

if [ -f "medsam_vit_b.pth" ]; then
    echo -e "${GREEN}✓ MedSAM: Ready ($(du -h medsam_vit_b.pth | cut -f1))${NC}"
else
    echo -e "${RED}✗ MedSAM: Not downloaded${NC}"
fi

if [ -d "CheXpert-v1.0-small" ]; then
    echo -e "${GREEN}✓ CheXpert: Ready ($(du -sh CheXpert-v1.0-small | cut -f1))${NC}"
else
    echo -e "${YELLOW}⚠ CheXpert: Not downloaded (requires registration)${NC}"
fi

if [ -d "mimic-cxr-jpg" ]; then
    echo -e "${GREEN}✓ MIMIC-CXR: Ready ($(du -sh mimic-cxr-jpg | cut -f1))${NC}"
else
    echo -e "${YELLOW}⚠ MIMIC-CXR: Not downloaded (requires credentialing)${NC}"
fi

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. If downloads complete, run:"
echo "   python scripts/preprocess_external_data.py"
echo ""
echo "2. Then train models with external pretraining:"
echo "   bash scripts/train_with_external_data.sh"
echo ""
echo "3. For MedSAM ROI extraction:"
echo "   python scripts/medsam_roi_extraction.py"
echo ""

echo "Download script completed!"
