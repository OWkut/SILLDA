import subprocess
import sys
import os

# Fonction pour créer un nouvel environnement Conda
def create_conda_env():
    env_name = "my_env"
    python_version = "3.8"

    # Vérifier si l'environnement existe déjà
    env_check = subprocess.run(
        ["conda", "env", "list"], capture_output=True, text=True
    )
    if env_name not in env_check.stdout:
        print(f"Création de l'environnement Conda '{env_name}' avec Python {python_version}...")
        subprocess.run(
            [
                "conda", "create", "-n", env_name, f"python={python_version}", "-y"
            ],
            check=True,
        )
    else:
        print(f"L'environnement '{env_name}' existe déjà.")

# Fonction pour installer les dépendances
def install_dependencies():
    env_name = "my_env"
    
    # Activer l'environnement et installer les dépendances
    print("Installation des dépendances dans l'environnement Conda...")
    subprocess.run(
        ["conda", "run", "-n", env_name, "pip", "install", "-r", "requirements.txt"],
        check=True,
    )

# Demander à l'utilisateur s'il souhaite créer un nouvel environnement ou utiliser l'existant
def ask_user():
    choice = input(
        "Souhaitez-vous créer un nouvel environnement Conda pour ce projet ? (o/n) : "
    ).lower()

    if choice == 'o':
        create_conda_env()
    elif choice == 'n':
        print("Vous avez choisi de conserver l'environnement actuel.")
    else:
        print("Réponse non valide. Le script va se terminer.")
        sys.exit(1)

# Exécuter les étapes ci-dessus
def main():
    ask_user()  # Demander à l'utilisateur s'il souhaite créer un nouvel environnement
    install_dependencies()  # Installer les dépendances dans l'environnement Conda
    print("Installation terminée.")

if __name__ == "__main__":
    main()
