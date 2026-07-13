import subprocess
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def run_command(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, shell=True)
    if result.returncode != 0:
        print(f"Error executing command: {cmd}")
        sys.exit(result.returncode)

def main():
    print("  RetinaAI Standalone Executable Packaging Script")
    print("  Mode: --onedir (low RAM, fast startup)")
    # Resolving Python interpreter
    python_exe = PROJECT_ROOT.parent / "retina_env" / "Scripts" / "python.exe"
    if not python_exe.exists():
        python_exe = Path(sys.executable)

    print(f"Using Python interpreter: {python_exe}")
    
    # Check if pyinstaller is installed
    try:
        subprocess.run([str(python_exe), "-c", "import PyInstaller"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("PyInstaller is already installed.")
    except subprocess.CalledProcessError:
        print("PyInstaller not found. Installing via pip...")
        run_command([str(python_exe), "-m", "pip", "install", "pyinstaller"])

    
    config_src = "configs/config.yaml"
    unet_ckpt_src = "experiments/exp_04/checkpoints/best.pth"
    cls_ckpt_src = "experiments/exp_04_cls_hybrid/checkpoints/best.pth"

    # Verify essential files exist before compiling
    for src in [config_src, unet_ckpt_src, cls_ckpt_src]:
        full_path = PROJECT_ROOT / src
        if not full_path.exists():
            print(f"CRITICAL ERROR: Required file does not exist: {full_path}")
            sys.exit(1)

    print("All required resource files verified.")

    # 3. Clean previous builds
    for folder in ["build", "dist"]:
        path = PROJECT_ROOT / folder
        if path.exists():
            print(f"Cleaning existing {folder} folder...")
            shutil.rmtree(path, ignore_errors=True)

    pyinstaller_exe = PROJECT_ROOT.parent / "retina_env" / "Scripts" / "pyinstaller.exe"
    if not pyinstaller_exe.exists():
        pyinstaller_exe = "pyinstaller"
    else:
        pyinstaller_exe = str(pyinstaller_exe)

    pyinstaller_cmd = [
        pyinstaller_exe,
        "--name=RetinaAI",
        "--onedir",           
        "--windowed",         
        "--noconfirm",        
        "--paths=.",
        "--exclude-module=matplotlib",
        "--exclude-module=pandas",
        "--exclude-module=tensorflow",
        "--exclude-module=IPython",
        "--exclude-module=ipywidgets",
        "--exclude-module=notebook",
        "--exclude-module=jupyter",
        "--exclude-module=seaborn",
        "--exclude-module=pytest",
        "--exclude-module=sphinx",
        "--exclude-module=docutils",
        # Bundle config and model checkpoints
        f"--add-data={config_src};configs",
        f"--add-data={unet_ckpt_src};experiments/exp_04/checkpoints",
        f"--add-data={cls_ckpt_src};experiments/exp_04_cls_hybrid/checkpoints",
        "app.py"
    ]

    print("\nStarting compilation process. This may take several minutes as PyTorch is large...")
    run_command(pyinstaller_cmd, cwd=str(PROJECT_ROOT))
    
    print("SUCCESS: Standalone executable built at:")
    print(f"  {PROJECT_ROOT / 'dist' / 'RetinaAI' / 'RetinaAI.exe'}")
    print("")
    print("To distribute, copy the entire 'dist/RetinaAI/' folder.")
    
if __name__ == "__main__":
    main()
