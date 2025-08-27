"""Setup script for MEA-Flow development."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔧 {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(e.stderr)
        return False


def main():
    """Set up MEA-Flow development environment."""
    print("🚀 Setting up MEA-Flow development environment...")
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: pyproject.toml not found. Run this script from the MEA-Flow root directory.")
        sys.exit(1)
    
    # Install uv if not available
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✅ uv is already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("📦 Installing uv...")
        if not run_command("curl -LsSf https://astral.sh/uv/install.sh | sh", "Installing uv"):
            print("❌ Failed to install uv. Please install manually: https://github.com/astral-sh/uv")
            sys.exit(1)
    
    # Create virtual environment
    if not run_command("uv venv", "Creating virtual environment"):
        sys.exit(1)
    
    # Install dependencies
    if not run_command("uv pip install -e .", "Installing MEA-Flow in development mode"):
        sys.exit(1)
    
    # Install optional dependencies
    if not run_command("uv pip install umap-learn", "Installing optional UMAP dependency"):
        print("⚠️ UMAP installation failed - UMAP embeddings will not be available")
    
    # Install development dependencies
    if not run_command("uv pip install jupyter ipykernel pytest black flake8", "Installing development tools"):
        print("⚠️ Some development tools failed to install")
    
    # Create kernel for Jupyter
    if not run_command("python -m ipykernel install --user --name mea-flow --display-name 'MEA-Flow'", "Setting up Jupyter kernel"):
        print("⚠️ Jupyter kernel setup failed")
    
    print("\n🎉 MEA-Flow setup completed!")
    print("\nTo activate the environment:")
    print("  source .venv/bin/activate  # Linux/macOS")
    print("  .venv\\Scripts\\activate     # Windows")
    print("\nTo run the tutorial:")
    print("  jupyter lab notebooks/01_mea_flow_tutorial.ipynb")
    print("\nTo test the installation:")
    print("  python -c 'import mea_flow; print(\"MEA-Flow ready!\")'")


if __name__ == "__main__":
    main()