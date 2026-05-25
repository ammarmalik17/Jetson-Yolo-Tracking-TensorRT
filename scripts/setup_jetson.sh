#!/bin/bash
# =============================================================================
# Jetson YOLO Tracking TensorRT - Setup & Validation Script
# =============================================================================
# This script validates and sets up the environment for running YOLO tracking
# on NVIDIA Jetson devices with TensorRT acceleration.
#
# Usage: bash scripts/setup_jetson.sh
# Run this on your Jetson device (NOT on Windows)
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_pass() {
    echo -e "${GREEN}✓ PASS:${NC} $1"
    ((PASS_COUNT++))
}

print_fail() {
    echo -e "${RED}✗ FAIL:${NC} $1"
    ((FAIL_COUNT++))
}

print_warn() {
    echo -e "${YELLOW}⚠ WARN:${NC} $1"
    ((WARN_COUNT++))
}

print_info() {
    echo -e "${BLUE}ℹ INFO:${NC} $1"
}

# =============================================================================
# System Information
# =============================================================================

print_header "System Information"

# Check if running on Jetson
if command -v jetson_release &> /dev/null; then
    print_info "Jetson Release Info:"
    jetson_release 2>/dev/null || print_warn "Could not retrieve Jetson release info"
else
    print_warn "jetson_release not installed (install with: sudo pip install jetson-stats)"
fi

# Check L4T version
if [ -f /etc/nv_tegra_release ]; then
    L4T_VERSION=$(head -n 1 /etc/nv_tegra_release | grep -oP 'R\d+' | head -1)
    print_pass "L4T Version: $L4T_VERSION"
else
    print_warn "Could not determine L4T version"
fi

# Check JetPack version
if command -v dpkg &> /dev/null; then
    JETPACK_VERSION=$(dpkg-query --showformat='${Version}' --show nvidia-l4t-core 2>/dev/null | cut -d'-' -f1 || echo "Unknown")
    print_info "JetPack Version: $JETPACK_VERSION"
else
    print_warn "Could not determine JetPack version"
fi

# Architecture check
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    print_pass "Architecture: $ARCH (Jetson/ARM64)"
else
    print_warn "Architecture: $ARCH (Expected aarch64 for Jetson)"
fi

# =============================================================================
# Power Mode & Performance
# =============================================================================

print_header "Power Mode & Performance"

# Check power mode
if command -v nvpmodel &> /dev/null; then
    POWER_MODE=$(sudo nvpmodel -q 2>/dev/null | grep "Power Mode" | awk -F' ' '{print $NF}' || echo "Unknown")
    print_info "Current Power Mode: $POWER_MODE"
    
    if [[ "$POWER_MODE" == *"MAX"* ]] || [[ "$POWER_MODE" == *"0"* ]]; then
        print_pass "Power mode is set to MAX performance"
    else
        print_warn "Power mode is not set to MAX. Run: sudo nvpmodel -m 0"
    fi
else
    print_warn "nvpmodel not found (may not be available on all Jetson devices)"
fi

# Check jetson_clocks
if command -v jetson_clocks &> /dev/null; then
    print_pass "jetson_clocks is available"
    print_info "To enable max clocks, run: sudo jetson_clocks"
else
    print_warn "jetson_clocks not found"
fi

# Check jtop
if command -v jtop &> /dev/null; then
    print_pass "jtop (jetson-stats) is installed"
else
    print_warn "jtop not installed. Install with: sudo pip install jetson-stats"
    print_info "jtop provides real-time monitoring of Jetson resources"
fi

# =============================================================================
# Python & Dependencies
# =============================================================================

print_header "Python & Dependencies"

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_pass "Python version: $PYTHON_VERSION"
else
    print_fail "Python3 not found! Install with: sudo apt install python3 python3-pip"
    exit 1
fi

# Check pip
if command -v pip3 &> /dev/null || command -v pip &> /dev/null; then
    print_pass "pip is installed"
else
    print_fail "pip not found! Install with: sudo apt install python3-pip"
    exit 1
fi

# Check virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    print_pass "Virtual environment is active: $VIRTUAL_ENV"
else
    print_warn "No virtual environment active (recommended for isolation)"
    print_info "Create one with: python3 -m venv .venv && source .venv/bin/activate"
fi

# =============================================================================
# Core Packages
# =============================================================================

print_header "Core Packages Check"

# Check Ultralytics
if python3 -c "import ultralytics; print(ultralytics.__version__)" &> /dev/null; then
    ULTRALYTICS_VERSION=$(python3 -c "import ultralytics; print(ultralytics.__version__)")
    print_pass "Ultralytics installed: v$ULTRALYTICS_VERSION"
else
    print_fail "Ultralytics not installed!"
    print_info "Install with: pip install ultralytics"
fi

# Check PyTorch
if python3 -c "import torch; print(torch.__version__)" &> /dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    CUDA_AVAILABLE=$(python3 -c "import torch; print('Yes' if torch.cuda.is_available() else 'No')")
    print_pass "PyTorch installed: v$TORCH_VERSION"
    print_info "CUDA Available: $CUDA_AVAILABLE"
    
    if [ "$CUDA_AVAILABLE" = "No" ]; then
        print_fail "CUDA is not available in PyTorch!"
        print_info "Reinstall PyTorch with CUDA support for your JetPack version"
    fi
else
    print_fail "PyTorch not installed!"
    print_info "Install from: https://docs.ultralytics.com/guides/nvidia-jetson"
fi

# Check TorchVision
if python3 -c "import torchvision; print(torchvision.__version__)" &> /dev/null; then
    TORCHVISION_VERSION=$(python3 -c "import torchvision; print(torchvision.__version__)")
    print_pass "TorchVision installed: v$TORCHVISION_VERSION"
else
    print_fail "TorchVision not installed!"
fi

# Check OpenCV
if python3 -c "import cv2; print(cv2.__version__)" &> /dev/null; then
    OPENCV_VERSION=$(python3 -c "import cv2; print(cv2.__version__)")
    print_pass "OpenCV installed: v$OPENCV_VERSION"
    
    # Check GStreamer support
    if python3 -c "import cv2; print(cv2.getBuildInformation())" 2>/dev/null | grep -i "GStreamer" | grep -q "YES"; then
        print_pass "OpenCV GStreamer support: Enabled"
    else
        print_warn "OpenCV GStreamer support: Not detected (required for Jetson camera)"
        print_info "Install system OpenCV: sudo apt install python3-opencv"
    fi
else
    print_fail "OpenCV not installed!"
    print_info "Install with: sudo apt install python3-opencv"
fi

# Check TensorRT
if python3 -c "import tensorrt; print(tensorrt.__version__)" &> /dev/null; then
    TRT_VERSION=$(python3 -c "import tensorrt; print(tensorrt.__version__)")
    print_pass "TensorRT Python bindings: v$TRT_VERSION"
else
    # TensorRT might be installed as system package
    if command -v trtexec &> /dev/null; then
        print_pass "TensorRT trtexec tool is available"
        print_warn "TensorRT Python bindings not detected (may still work via Ultralytics)"
    else
        print_fail "TensorRT not found!"
        print_info "TensorRT should be included with JetPack. Re-flash if missing."
    fi
fi

# =============================================================================
# Model Files
# =============================================================================

print_header "Model Files Check"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check for models directory
if [ -d "$PROJECT_DIR/models" ]; then
    print_pass "models/ directory exists"
    
    # Count model files
    MODEL_COUNT=$(find "$PROJECT_DIR/models" -type f \( -name "*.engine" -o -name "*.pt" -o -name "*.onnx" \) 2>/dev/null | wc -l)
    print_info "Found $MODEL_COUNT model file(s) in models/"
    
    # List models
    if [ "$MODEL_COUNT" -gt 0 ]; then
        print_info "Available models:"
        find "$PROJECT_DIR/models" -type f \( -name "*.engine" -o -name "*.pt" -o -name "*.onnx" \) -exec basename {} \; 2>/dev/null | while read -r model; do
            echo "   - $model"
        done
    else
        print_warn "No model files found in models/"
        print_info "Export a model with: python tools/export_tensorrt_engine.py"
    fi
else
    print_fail "models/ directory not found!"
fi

# =============================================================================
# Camera Test (Optional)
# =============================================================================

print_header "Camera Test (Optional)"

print_info "Testing Jetson ARGUS camera access..."

# Run camera test script if it exists
if [ -f "$SCRIPT_DIR/jetson_camera_test.py" ]; then
    print_info "Running camera diagnostics..."
    python3 "$SCRIPT_DIR/jetson_camera_test.py" || print_warn "Camera test failed (camera may not be connected)"
else
    print_warn "jetson_camera_test.py not found"
fi

# =============================================================================
# Quick Inference Test
# =============================================================================

print_header "Quick Inference Test"

# Find first available engine file
ENGINE_FILE=$(find "$PROJECT_DIR/models" -name "*.engine" -type f 2>/dev/null | head -1)

if [ -n "$ENGINE_FILE" ]; then
    print_info "Testing inference with: $(basename "$ENGINE_FILE")"
    
    # Quick test
    if python3 -c "
from ultralytics import YOLO
import sys

try:
    model = YOLO('$ENGINE_FILE', task='detect')
    print('✓ Model loaded successfully')
    sys.exit(0)
except Exception as e:
    print(f'✗ Model loading failed: {e}')
    sys.exit(1)
" 2>&1; then
        print_pass "TensorRT engine loads successfully"
    else
        print_fail "Failed to load TensorRT engine"
        print_info "Re-export the engine: python tools/export_tensorrt_engine.py"
    fi
else
    print_warn "No .engine file found for inference test"
    print_info "Export one with: python tools/export_tensorrt_engine.py"
fi

# =============================================================================
# Performance Recommendations
# =============================================================================

print_header "Performance Recommendations"

print_info "For optimal performance on Jetson:"
echo ""
echo "  1. Enable MAX power mode:"
echo "     sudo nvpmodel -m 0"
echo ""
echo "  2. Lock CPU/GPU clocks:"
echo "     sudo jetson_clocks"
echo ""
echo "  3. Monitor performance:"
echo "     sudo jtop"
echo ""
echo "  4. Use FP16 TensorRT engines (already configured)"
echo ""
echo "  5. For Jetson AI Lab PyPI mirror (faster installs):"
echo "     pip install --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu130/+simple/ -r requirements.txt"
echo ""

# =============================================================================
# Summary
# =============================================================================

print_header "Setup Summary"

TOTAL=$((PASS_COUNT + FAIL_COUNT + WARN_COUNT))
echo -e "Total checks: $TOTAL"
echo -e "${GREEN}Passed: $PASS_COUNT${NC}"
echo -e "${RED}Failed: $FAIL_COUNT${NC}"
echo -e "${YELLOW}Warnings: $WARN_COUNT${NC}"
echo ""

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}  ✓ Setup looks good! You're ready to run YOLO tracking.${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Export or place a model in models/"
    echo "  2. Run: python scripts/yolo_track_lock.py --source 0 --enable-lock"
    echo ""
else
    echo -e "${RED}============================================================${NC}"
    echo -e "${RED}  ✗ $FAIL_COUNT critical issue(s) found. Please fix before proceeding.${NC}"
    echo -e "${RED}============================================================${NC}"
    echo ""
    exit 1
fi

if [ "$WARN_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}Note: $WARN_COUNT warning(s) detected. These are non-critical but recommended to address.${NC}"
    echo ""
fi

print_info "For troubleshooting, see: https://docs.ultralytics.com/guides/nvidia-jetson/"
print_info "Report issues: https://github.com/ammarmalik17/Yolo-TensorRT-Tracking-and-Jetson-Camera-Toolkit/issues"
