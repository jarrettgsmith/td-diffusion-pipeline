"""
Config Loader for SDXL TouchDesigner Integration
Loads config.yaml with conda environment and model settings.
"""
import os
import sys
import yaml
from pathlib import Path


def find_project_root():
    """Find project root by looking for config.yaml"""
    
    # Method 1: TouchDesigner project folder (if available)
    try:
        if 'project' in globals() and hasattr(project, 'folder'):
            project_dir = Path(project.folder)
            config_path = project_dir / "config.yaml"
            if config_path.exists():
                print(f"[Config] Found via TouchDesigner project: {project_dir}")
                return project_dir
    except:
        pass
    
    # Method 2: Use __file__ if available
    try:
        current = Path(__file__).resolve().parent
        while current != current.parent:
            config_path = current / "config.yaml"
            if config_path.exists():
                print(f"[Config] Found via __file__: {current}")
                return current
            current = current.parent
    except:
        pass
    
    # Method 3: Check current working directory
    try:
        current = Path.cwd()
        for i in range(5):
            config_path = current / "config.yaml"
            if config_path.exists():
                print(f"[Config] Found via cwd: {current}")
                return current
            current = current.parent
    except:
        pass
    
    # Method 4: Check sys.path entries
    for path_entry in sys.path:
        try:
            check_path = Path(path_entry).resolve()
            config_path = check_path / "config.yaml"
            if config_path.exists():
                print(f"[Config] Found via sys.path: {check_path}")
                return check_path
        except:
            continue
    
    raise FileNotFoundError("Could not find config.yaml in project hierarchy")


def get_config():
    """Load configuration from config.yaml"""
    project_root = find_project_root()
    config_path = project_root / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Store the project root in the config for path resolution
    config['_project_root'] = project_root
    
    print(f"[Config] Loaded configuration from: {config_path}")
    return config


def get_python():
    """Get the Python executable path"""
    config = get_config()
    project_root = config['_project_root']
    
    # Build path relative to project root
    venv_name = config['python'].get('venv', '.venv')
    venv_path = project_root / venv_name
    
    # Determine Python executable based on OS
    if os.name == 'nt':  # Windows
        python_path = venv_path / 'Scripts' / 'python.exe'
    else:  # Unix-like
        python_path = venv_path / 'bin' / 'python'
    
    if not python_path.exists():
        raise FileNotFoundError(f"Python not found: {python_path}")
    
    return str(python_path)

# Backward compatibility
def get_conda_python():
    """Deprecated: Use get_python() instead"""
    return get_python()


def get_server_script():
    """Get the full path to the server script"""
    config = get_config()
    project_root = config['_project_root']
    
    # Get script path from config, default to spout_diffusion_server.py
    script_name = config.get('paths', {}).get('server_script', 'spout_diffusion_server.py')
    script_path = project_root / script_name
    
    if not script_path.exists():
        raise FileNotFoundError(f"Server script not found: {script_path}")
    
    return str(script_path)


def get_project_root():
    """Get the project root directory (where config.yaml is located)"""
    config = get_config()
    return config['_project_root']


if __name__ == "__main__":
    # Test the config loader
    try:
        config = get_config()
        print("Config loaded successfully")
        print(f"Project root: {config['_project_root']}")
        print(f"Python executable: {get_python()}")
        print(f"Server script: {get_server_script()}")
        print(f"Diffusion model: {config['diffusion']['model_id']}")
    except Exception as e:
        print(f"Config error: {e}")