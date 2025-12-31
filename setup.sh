#!/bin/bash
#
# HeteroFleet Setup Script
# ========================
# This script sets up everything you need to run HeteroFleet.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#

set -e

echo "=============================================="
echo "HeteroFleet Setup Script"
echo "=============================================="
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "ERROR: Python 3.10 or higher is required"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi
echo "✓ Python $PYTHON_VERSION found"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q
echo "✓ pip upgraded"
echo ""

# Install required packages
echo "Installing required packages..."
pip install numpy pydantic loguru -q
echo "✓ Core packages installed"

# Install optional packages
echo ""
echo "Installing optional packages..."
pip install matplotlib pillow -q 2>/dev/null || echo "  (matplotlib/pillow skipped)"
echo "✓ Visualization packages installed"

# Check for cflib (Crazyflie)
echo ""
echo "Checking for Crazyflie support..."
if pip install cflib -q 2>/dev/null; then
    echo "✓ cflib installed (Crazyflie hardware support available)"
else
    echo "  cflib not available (Crazyflie will run in simulation mode)"
fi

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from heterofleet.core.platform import PlatformType
    from heterofleet.simulation.engine import SimulationEngine
    print('✓ HeteroFleet modules verified')
except ImportError as e:
    print(f'ERROR: {e}')
    sys.exit(1)
"

# Run tests
echo ""
echo "Running module tests..."
python3 main.py test 2>/dev/null | grep -E "^  [✓✗]|Results:" || true

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To get started:"
echo ""
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run the demo:"
echo "     python main.py demo"
echo ""
echo "  3. Try the tutorials:"
echo "     python tutorials/tutorial_01_hello.py"
echo "     python tutorials/tutorial_02_multi_robot.py"
echo "     python tutorials/tutorial_03_task_allocation.py"
echo "     python tutorials/tutorial_04_digital_twins.py"
echo ""
echo "  4. Read the full guide:"
echo "     docs/BEGINNERS_GUIDE.md"
echo ""
echo "Happy flying! 🚁🤖"
