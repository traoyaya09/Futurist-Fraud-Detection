"""
Install CLIP from GitHub (the correct way)
"""
import subprocess
import sys

def main():
    print("=" * 80)
    print("Installing CLIP from GitHub...")
    print("=" * 80)
    print()
    
    try:
        print("Running: pip install git+https://github.com/openai/CLIP.git")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print()
        print("=" * 80)
        print("[SUCCESS] CLIP installed successfully!")
        print("=" * 80)
        print()
        print("Next step:")
        print("  python train_all_models.py --skip-checks --skip-images")
        return 0
    except Exception as e:
        print()
        print("=" * 80)
        print(f"[FAILED] Could not install CLIP: {e}")
        print("=" * 80)
        print()
        print("Try manually:")
        print("  pip install git+https://github.com/openai/CLIP.git")
        return 1

if __name__ == "__main__":
    sys.exit(main())