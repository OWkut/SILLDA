import subprocess
import os

def create_conda_env(env_name):
    # Crée un nouvel environnement Conda avec Python 3.8 si l'environnement n'existe pas
    print(f"Création de l'environnement Conda '{env_name}' avec Python 3.8...")
    subprocess.run(["conda", "create", "-n", env_name, "python=3.8", "-y"])

def install_dependencies(env_name):
    # Utilise conda run pour activer l'environnement et installer les dépendances
    print(f"Installation des dépendances dans l'environnement Conda '{env_name}'...")
    subprocess.run(
        ["conda", "run", "-n", env_name, "pip", "install", "-r", "requirements.txt"],
        check=True
    )

def main():
    # Demande à l'utilisateur s'il veut créer un nouvel environnement
    env_name = "SILLDA"
    create_new_env = input("Souhaitez-vous créer un nouvel environnement Conda pour ce projet ? (o/n) : ").strip().lower()

    if create_new_env == "o":
        if not os.path.exists(f"/home/theob/anaconda3/envs/{env_name}"):
            create_conda_env(env_name)
        else:
            print(f"L'environnement '{env_name}' existe déjà.")
    
    print(f"Activation de l'environnement Conda '{env_name}' et installation des dépendances...")
    
    # Utilisation de conda run pour activer l'environnement et installer les dépendances
    install_dependencies(env_name)
    print("Installation terminée.")

if __name__ == "__main__":
    main()
