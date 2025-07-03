#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run script for Synthetic Data Generator.
This script provides a simple way to launch the generator from the command line.
"""

import sys
import traceback
from synthetic_data_gen.main import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation canceled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nIf this is unexpected, please report this issue.")
        choice = input("Show detailed error? (y/n): ").strip().lower()
        if choice == 'y':
            traceback.print_exc()
        sys.exit(1)
