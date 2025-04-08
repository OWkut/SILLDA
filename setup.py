import subprocess
import sys
import os

def install_requirements():
    print("🔧 Installation des dépendances...")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Toutes les dépendances sont installées.")
    except subprocess.CalledProcessError as e:
        print("❌ Une erreur est survenue lors de l'installation des dépendances.")
        sys.exit(1)

def main():
    print("Lancement du script principal...")
    os.system(f"{sys.executable} app.py")

if __name__ == "__main__":
    install_requirements()
    main()