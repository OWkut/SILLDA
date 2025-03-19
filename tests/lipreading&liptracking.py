# Désactiver les erreurs liées à Tensorflow
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Réduit les logs TensorFlow
import tensorflow as tf
tf.get_logger().setLevel("ERROR")  # Masque les messages supplémentaires

# Permet de charger des modules depuis nimporte quel endroit du projet
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # Récupérer le chemin absolu du dossier racine du projet
sys.path.append(project_root) # Ajouter ce chemin au sys.path

from src.lip_reading.LipReadingModel import LipReadingModel


model = LipReadingModel()
video_path = "data/raw/lip_reading_dlib/bbaf2n.mpg"
frames = model.load_video_with_liptracking(video_path)
print(model.predict_text(frames))

# video_path = "data/raw/lip_reading_dlib/bbaf2n-bbaf3s.mp4"
# frames = model.load_long_video_with_liptracking(video_path)
# print(model.predict_long_text(frames))