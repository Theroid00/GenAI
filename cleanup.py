#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cleanup Script
-------------
This script cleans up temporary and test files to prepare the codebase for sharing.
"""

import os
import re
import glob
import shutil

def clean_directory():
    """Clean up temporary and test files"""
    print("Cleaning up the repository...")
    
    # List of file patterns to remove
    patterns_to_remove = [
        "test_*.csv",
        "test_*.json",
        "final_*.csv",
        "*_temp.*",
        "*_tmp.*",
        "*_copy.*",
        "*_backup.*"
    ]
    
    # Count of removed files
    removed_count = 0
    
    # Remove files matching patterns
    for pattern in patterns_to_remove:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"Error removing {file_path}: {str(e)}")
    
    # Remove __pycache__ directories
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__' or dir_name.endswith('.egg-info'):
                try:
                    dir_path = os.path.join(root, dir_name)
                    shutil.rmtree(dir_path)
                    print(f"Removed directory: {dir_path}")
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing directory {dir_path}: {str(e)}")
    
    print(f"\nCleanup complete! Removed {removed_count} files and directories.")
    print("Repository is now clean and ready for sharing.")

if __name__ == "__main__":
    clean_directory()
