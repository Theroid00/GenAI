#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for the Synthetic Data Generator.
This script creates a virtual environment and installs all required dependencies.

Usage:
    ./setup_env.py         # On Linux/Mac (if script is executable)
    python setup_env.py    # On any platform
"""

import os
import subprocess
import sys
import platform
import time
from pathlib import Path
from typing import List, Tuple

def check_python_version() -> Tuple[bool, str]:
    """
    Check if the Python version meets the minimum requirements.
    
    Returns:
        Tuple of (meets_requirements, message)
    """
    min_version = (3, 6)
    current_version = sys.version_info[:2]
    
    if current_version >= min_version:
        return True, f"Python version {sys.version.split()[0]} is compatible."
    else:
        return False, (f"Python version {sys.version.split()[0]} is not compatible. "
                     f"Minimum required version is {min_version[0]}.{min_version[1]}.")

def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """
    Run a shell command and handle errors.
    
    Args:
        cmd: List of command components
        description: Description of what the command does
        
    Returns:
        Tuple of (success, message)
    """
    print(f"{description}...")
    try:
        process = subprocess.run(
            cmd, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        return True, process.stdout
    except subprocess.CalledProcessError as e:
        error_msg = f"Error: {e}\n{e.stderr}"
        return False, error_msg
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def create_virtual_env(venv_dir: Path, python_cmd: str) -> Tuple[bool, str]:
    """
    Create a virtual environment.
    
    Args:
        venv_dir: Path to create the virtual environment in
        python_cmd: Python executable to use
        
    Returns:
        Tuple of (success, message)
    """
    return run_command(
        [python_cmd, "-m", "venv", str(venv_dir)],
        "Creating virtual environment"
    )

def get_pip_command(venv_dir: Path) -> str:
    """
    Get the pip command for the virtual environment.
    
    Args:
        venv_dir: Path to the virtual environment
        
    Returns:
        Path to the pip executable
    """
    if platform.system() == "Windows":
        return str(venv_dir / "Scripts" / "pip")
    else:
        return str(venv_dir / "bin" / "pip")

def upgrade_pip(pip_cmd: str) -> Tuple[bool, str]:
    """
    Upgrade pip in the virtual environment.
    
    Args:
        pip_cmd: Path to the pip executable
        
    Returns:
        Tuple of (success, message)
    """
    return run_command(
        [pip_cmd, "install", "--upgrade", "pip"],
        "Upgrading pip in the virtual environment"
    )

def install_dependencies(pip_cmd: str, requirements_path: Path) -> Tuple[bool, str]:
    """
    Install dependencies from requirements.txt.
    
    Args:
        pip_cmd: Path to the pip executable
        requirements_path: Path to the requirements.txt file
        
    Returns:
        Tuple of (success, message)
    """
    return run_command(
        [pip_cmd, "install", "-r", str(requirements_path)],
        "Installing dependencies"
    )

def install_package(pip_cmd: str, project_dir: Path) -> Tuple[bool, str]:
    """
    Install the package in development mode.
    
    Args:
        pip_cmd: Path to the pip executable
        project_dir: Path to the project directory
        
    Returns:
        Tuple of (success, message)
    """
    return run_command(
        [pip_cmd, "install", "-e", "."],
        "Installing package in development mode"
    )

def main() -> int:
    """Main function to set up the environment"""
    start_time = time.time()
    
    print("=" * 80)
    print("SYNTHETIC DATA GENERATOR - SETUP")
    print("=" * 80)
    
    # Check Python version
    python_compatible, python_message = check_python_version()
    print(python_message)
    
    if not python_compatible:
        print("Setup aborted due to incompatible Python version.")
        return 1
    
    # Determine the project directory
    project_dir = Path(__file__).parent.absolute()
    print(f"Project directory: {project_dir}")
    
    # Check if Python is available
    python_cmd = sys.executable
    print(f"Using Python: {python_cmd}")
    
    # Check if requirements.txt exists
    requirements_path = project_dir / "requirements.txt"
    if not requirements_path.exists():
        print(f"Error: requirements.txt not found at {requirements_path}")
        return 1
    
    # Check if setup.py exists
    setup_path = project_dir / "setup.py"
    if not setup_path.exists():
        print(f"Error: setup.py not found at {setup_path}")
        return 1
    
    # Check if virtual environment exists
    venv_dir = project_dir / ".venv"
    venv_exists = venv_dir.exists()
    
    if venv_exists:
        print("\nVirtual environment already exists.")
        recreate = input("Do you want to recreate it? (yes/no) [no]: ").strip().lower()
        if recreate == "yes":
            print("Removing existing virtual environment...")
            try:
                # Remove the venv directory
                if platform.system() == "Windows":
                    subprocess.run(["rmdir", "/S", "/Q", str(venv_dir)], check=True)
                else:
                    subprocess.run(["rm", "-rf", str(venv_dir)], check=True)
                venv_exists = False
            except subprocess.CalledProcessError as e:
                print(f"Error removing virtual environment: {e}")
                return 1
            except Exception as e:
                print(f"Unexpected error removing virtual environment: {str(e)}")
                return 1
    
    # Create virtual environment if it doesn't exist
    if not venv_exists:
        success, message = create_virtual_env(venv_dir, python_cmd)
        if not success:
            print(message)
            return 1
    
    # Determine the pip command based on the operating system
    pip_cmd = get_pip_command(venv_dir)
    
    # Upgrade pip in the virtual environment
    success, message = upgrade_pip(pip_cmd)
    if not success:
        print(message)
        print("Warning: Failed to upgrade pip, but continuing with setup...")
    
    # Install dependencies
    success, message = install_dependencies(pip_cmd, requirements_path)
    if not success:
        print(message)
        return 1
    
    # Install the package in development mode
    success, message = install_package(pip_cmd, project_dir)
    if not success:
        print(message)
        return 1
    
    # Calculate setup time
    end_time = time.time()
    setup_time = end_time - start_time
    
    # Installation complete
    print("\n" + "=" * 80)
    print("INSTALLATION COMPLETE")
    print(f"Setup completed in {setup_time:.1f} seconds")
    print("=" * 80)
    
    # Activation instructions
    print("\nTo activate the virtual environment:")
    if platform.system() == "Windows":
        print(f"    {venv_dir}\\Scripts\\activate")
    else:
        print(f"    source {venv_dir}/bin/activate")
    
    # Running instructions
    print("\nTo run the generator:")
    print("    python run_generator.py")
    
    # Optional SDV installation
    print("\nFor model-based generation (Mode 1), you may want to install the SDV package:")
    if platform.system() == "Windows":
        print(f"    {venv_dir}\\Scripts\\pip install sdv")
    else:
        print(f"    {venv_dir}/bin/pip install sdv")
    
    print("\nFor detailed documentation, see README.md and MODULARIZATION.md")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)
