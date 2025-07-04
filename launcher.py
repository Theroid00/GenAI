#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Synthetic Data Generator Launcher
--------------------------------
A simple launcher script for the Synthetic Data Generator.
"""

import os
import sys
import argparse
import textwrap
import subprocess
from pathlib import Path

# Import version from the package
try:
    from synthetic_data_gen import __version__
except ImportError:
    __version__ = "Unknown"

def display_intro():
    """Display an introduction to the Synthetic Data Generator"""
    intro = f"""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                   SYNTHETIC DATA GENERATOR v{__version__:<8}               ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    A comprehensive tool for generating high-quality synthetic data
    with built-in validation and customizable templates.
    
    This launcher will help you get started with the synthetic data generator.
    """
    print(textwrap.dedent(intro))

def display_options():
    """Display the available options"""
    options = """
    Available Commands:
    
    1. Generate data using a template
       - Quick and easy way to generate synthetic data
    
    2. Create a custom template
       - Interactive template creation
    
    3. View available templates
       - See what templates are available
    
    4. Generate data from a schema file
       - Use your own JSON schema
    
    5. Generate data from a sample CSV
       - Learn from existing data
    
    6. Launch interactive mode
       - Generate data interactively
    
    7. Check dependencies
       - Verify all required packages are installed
    
    8. View documentation
       - Open README.md in an editor
    
    0. Exit
    """
    print(textwrap.dedent(options))

def run_command(command):
    """Run a command and handle errors"""
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
    except KeyboardInterrupt:
        print("\nOperation cancelled.")

def generate_from_template():
    """Generate data using a template"""
    # Get list of templates
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    if not os.path.exists(templates_dir):
        print("No templates directory found. Create templates first.")
        return
        
    template_files = [f.replace('.json', '') for f in os.listdir(templates_dir) if f.endswith('.json')]
    
    if not template_files:
        print("No templates found. Create templates first.")
        return
    
    print("\nAvailable templates:")
    for i, template in enumerate(template_files, 1):
        print(f"  {i}. {template}")
    
    try:
        choice = int(input("\nSelect a template (number): "))
        if choice < 1 or choice > len(template_files):
            print("Invalid choice.")
            return
        
        template_name = template_files[choice-1]
        rows = input("Number of rows to generate [100]: ").strip() or "100"
        output = input(f"Output filename [{template_name}_data.csv]: ").strip() or f"{template_name}_data.csv"
        visualize = input("Visualize data distributions? (y/n) [n]: ").strip().lower() or "n"
        
        print(f"\nGenerating data using template: {template_name}")
        command = f"python synthetic_data_cli.py --template {template_name} --rows {rows} --output {output}"
        
        if visualize.startswith('y'):
            command += " --visualize"
            
        run_command(command)
    except ValueError:
        print("Invalid input.")
    except KeyboardInterrupt:
        print("\nOperation cancelled.")

def create_template():
    """Create a custom template"""
    try:
        name = input("Enter template name: ").strip()
        if not name:
            print("Template name is required.")
            return
        
        print(f"\nCreating template: {name}")
        command = f"python synthetic_data_cli.py --create-template {name}"
        run_command(command)
    except KeyboardInterrupt:
        print("\nOperation cancelled.")

def view_templates():
    """View available templates"""
    command = "python synthetic_data_cli.py --list-templates"
    run_command(command)
    
    # Ask if user wants to see details of a specific template
    try:
        view_details = input("\nDo you want to see details of a specific template? (y/n): ").strip().lower()
        if view_details.startswith('y'):
            template_name = input("Enter template name: ").strip()
            if template_name:
                command = f"python synthetic_data_cli.py --template-info {template_name}"
                run_command(command)
    except KeyboardInterrupt:
        print("\nOperation cancelled.")

def generate_from_schema():
    """Generate data from a schema file"""
    try:
        schema_file = input("Enter schema JSON file path: ").strip()
        if not os.path.exists(schema_file):
            print(f"File not found: {schema_file}")
            return
        
        rows = input("Number of rows to generate [100]: ").strip() or "100"
        output = input("Output filename [schema_data.csv]: ").strip() or "schema_data.csv"
        visualize = input("Visualize data distributions? (y/n) [n]: ").strip().lower() or "n"
        
        print(f"\nGenerating data from schema: {schema_file}")
        command = f"python synthetic_data_cli.py --schema {schema_file} --rows {rows} --output {output}"
        
        if visualize.startswith('y'):
            command += " --visualize"
            
        run_command(command)
    except KeyboardInterrupt:
        print("\nOperation cancelled.")

def generate_from_csv():
    """Generate data from a sample CSV file"""
    try:
        csv_file = input("Enter sample CSV file path: ").strip()
        if not os.path.exists(csv_file):
            print(f"File not found: {csv_file}")
            return
        
        rows = input("Number of rows to generate [100]: ").strip() or "100"
        output = input("Output filename [synthetic_data.csv]: ").strip() or "synthetic_data.csv"
        model = input("Model type (gaussian/ctgan) [gaussian]: ").strip().lower() or "gaussian"
        visualize = input("Visualize data distributions? (y/n) [n]: ").strip().lower() or "n"
        
        if model not in ["gaussian", "ctgan"]:
            print("Invalid model type. Using 'gaussian'.")
            model = "gaussian"
        
        print(f"\nGenerating data from CSV: {csv_file}")
        command = f"python synthetic_data_cli.py --sample {csv_file} --rows {rows} --output {output} --model {model}"
        
        if visualize.startswith('y'):
            command += " --visualize"
            
        run_command(command)
    except KeyboardInterrupt:
        print("\nOperation cancelled.")

def launch_interactive_mode():
    """Launch the interactive mode"""
    interactive_path = Path(__file__).parent / "interactive_mode.py"
    
    if not interactive_path.exists():
        print("Creating interactive mode script...")
        with open(interactive_path, 'w') as f:
            f.write("""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

\"\"\"
Interactive Mode for Synthetic Data Generator
---------------------------------------------
This script provides an interactive interface for generating synthetic data.
\"\"\"

import os
import sys
from pathlib import Path

# Import from the synthetic_data_gen package
from synthetic_data_gen import SyntheticDataGenerator

def main():
    \"\"\"Main function for interactive mode\"\"\"
    print("\\nWelcome to Interactive Mode!")
    print("This mode allows you to create a schema and generate data interactively.\\n")
    
    # Create generator instance
    generator = SyntheticDataGenerator()
    
    try:
        # Launch interactive mode
        generator.interactive_mode()
    except KeyboardInterrupt:
        print("\\nInteractive mode cancelled.")
    except Exception as e:
        print(f"\\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
""")
    
    print("\nLaunching interactive mode...")
    command = "python interactive_mode.py"
    run_command(command)

def check_dependencies():
    """Check all dependencies"""
    print("\nChecking dependencies...")
    command = "python synthetic_data_cli.py --check-deps"
    run_command(command)

def view_documentation():
    """View the documentation"""
    readme_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
    
    if not os.path.exists(readme_file):
        readme_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FINAL_README.md")
        if not os.path.exists(readme_file):
            print("README.md not found.")
            return
    
    # Try to determine the best way to open the file
    if sys.platform.startswith('linux'):
        # Try different text editors common on Linux
        editors = ['xdg-open', 'nano', 'vim', 'less', 'cat']
        for editor in editors:
            try:
                subprocess.run(f"which {editor}", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if editor == 'xdg-open':
                    run_command(f"{editor} {readme_file}")
                else:
                    run_command(f"{editor} {readme_file}")
                break
            except subprocess.CalledProcessError:
                continue
    elif sys.platform == 'darwin':  # macOS
        run_command(f"open {readme_file}")
    elif sys.platform == 'win32':  # Windows
        run_command(f"notepad {readme_file}")
    else:
        # Fallback: print the content of the file
        print("\nREADME.md Contents:")
        print("=" * 80)
        with open(readme_file, 'r') as f:
            print(f.read())

def main():
    """Main function for the launcher"""
    display_intro()
    
    while True:
        display_options()
        
        try:
            choice = input("\nEnter your choice (0-8): ").strip()
            
            if choice == '0':
                print("\nExiting. Goodbye!")
                break
            elif choice == '1':
                generate_from_template()
            elif choice == '2':
                create_template()
            elif choice == '3':
                view_templates()
            elif choice == '4':
                generate_from_schema()
            elif choice == '5':
                generate_from_csv()
            elif choice == '6':
                launch_interactive_mode()
            elif choice == '7':
                check_dependencies()
            elif choice == '8':
                view_documentation()
            else:
                print("Invalid choice. Please enter a number between 0 and 8.")
            
            # Pause before showing menu again
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\nExiting. Goodbye!")
            break

if __name__ == "__main__":
    main()
