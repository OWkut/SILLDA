import subprocess
import os

def create_conda_env(env_name):
    # Crée un nouvel environnement Conda avec Python 3.8 si l'environnement n'existe pas
    print(f"Création de l'environnement Conda '{env_name}' avec Python 3.8...")
    subprocess.run(["conda", "create", "-n", env_name, "python=3.8", "-y"])

def install_dependencies(env_name=None):
    # Si env_name est None, on utilise l'environnement actif actuel
    if env_name:
        print(f"Installation des dépendances dans l'environnement Conda '{env_name}'...")
        subprocess.run(
            ["conda", "run", "-n", env_name, "pip", "install", "-r", "requirements.txt"],
            check=True
        )
    else:
        print("Installation des dépendances dans l'environnement Conda actuel...")
        subprocess.run(
            ["pip", "install", "-r", "requirements.txt"],
            check=True
        )

def main():
    # Demande à l'utilisateur s'il veut créer un nouvel environnement
    create_new_env = input("Souhaitez-vous créer un nouvel environnement Conda pour ce projet ? (o/n) : ").strip().lower()

    if create_new_env == "o":
        env_name = input("Entrez le nom de l'environnement Conda que vous souhaitez créer : ").strip()
        
        # Vérifie si l'environnement existe déjà
        if os.path.exists(f"/home/theob/anaconda3/envs/{env_name}"):
            print(f"L'environnement '{env_name}' existe déjà.")
        else:
            create_conda_env(env_name)
        
        # Installer les dépendances dans le nouvel environnement
        install_dependencies(env_name)
    elif create_new_env == "n":
        # Installer les dépendances dans l'environnement actuel
        install_dependencies()
    else:
        print("Option non valide. Veuillez répondre par 'o' ou 'n'.")

    print("Installation terminée.")

if __name__ == "__main__":
    main()
