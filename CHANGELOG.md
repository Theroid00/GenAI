# Changelog

### Changelog

## v1.1.0 - 2025-07-03

### Added
- GitHub repository setup
- Added improved .gitignore file
- Created example_data directory for sample files

### Changed
- Updated setup_env.py with better error handling and timing information
- Updated README.md with more detailed usage instructions
- Improved documentation organization

## v1.0.1 - 2023-06-20

### Added
- `setup_env.py` script for automated environment setup
- Comprehensive error handling in all modules
- Added requirements.txt file for dependency management
- Updated README.md with detailed installation instructions

### Fixed
- Fixed input() bug in mode_1_generate_from_csv() that could cause KeyboardInterrupt
- Improved error handling throughout the code
- Enhanced CSV loading and saving with better error recovery

### Changed
- Improved project structure with proper modularization
- Enhanced documentation with MODULARIZATION.md
- Refactored code for better maintainability and readability3

### Added
- LICENSE file with MIT license
- Improved .gitignore file for better GitHub integration
- Example data in example_data/ directory

### Changed
- Cleaned up project structure for GitHub release
- Improved MANIFEST.in to ensure all necessary files are included in distribution

### Removed
- Original monolithic script (synthetic_data_generator.py)
- Temporary log files

## v1.0.1 - 2023-06-20

### Added
- `setup_env.py` script for automated environment setup
- Comprehensive error handling in all modules
- Added requirements.txt file for dependency management
- Updated README.md with detailed installation instructions

### Fixed
- Fixed input() bug in mode_1_generate_from_csv() that could cause KeyboardInterrupt
- Improved error handling throughout the code
- Enhanced CSV loading and saving with better error recovery

### Changed
- Improved project structure with proper modularization
- Enhanced documentation with MODULARIZATION.md
- Refactored code for better maintainability and readability

## v1.0.0 - 2023-06-01

### Added
- Initial release of the Synthetic Data Generator
- Three operational modes:
  - Mode 1: Generate from CSV sample
  - Mode 2: Generate interactively
  - Mode 3: Generate from saved schema
- Support for various data types and relationships
- Visualization and validation capabilities
