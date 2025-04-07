import numpy as np
import cv2
import dlib

class LipTracking:
    def __init__(self):
        # Initialisation des détecteurs et prédicteurs de dlib
        predictor_path = 'core/models/lip_reading_dlib/shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.width_crop_max = 0
        self.height_crop_max = 0
        self.target_aspect_ratio = 140 / 46  # Rapport d'aspect attendu par le modèle (140:46)

    def process_frame(self, frame):
        # Détection des visages dans la frame
        detections = self.detector(frame, 1)
        activation = 0
        lip_status = "Non détecté"
        lip_coordinates = None  # Initialisation des coordonnées de la bouche

        if len(detections) > 0:
            for k, d in enumerate(detections):
                # Prédiction des points du visage
                shape = self.predictor(frame, d)
                marks = np.zeros((2, 20))
                co = 0

                # Extraction des points de la bouche
                for ii in range(48, 68):
                    X = shape.part(ii)
                    marks[0, co] = X.x
                    marks[1, co] = X.y
                    co += 1

                # Calcul des points extrêmes de la bouche
                X_left, Y_left, X_right, Y_right = [int(np.amin(marks, axis=1)[0]), int(np.amin(marks, axis=1)[1]),
                                                    int(np.amax(marks, axis=1)[0]), int(np.amax(marks, axis=1)[1])]

                # Calcul du centre de la bouche
                X_center = (X_left + X_right) / 2.0
                Y_center = (Y_left + Y_right) / 2.0

                # Définition d'une bordure dynamique autour de la bouche
                border_width = int((X_right - X_left) * 0.55)  # Bordure de 50% de la largeur de la bouche
                border_height = int((Y_right - Y_left) * 0.55)  # Bordure de 50% de la hauteur de la bouche
                border = max(border_width, border_height)  # Utiliser la plus grande bordure

                X_left_new = X_left - border
                Y_left_new = Y_left - border
                X_right_new = X_right + border
                Y_right_new = Y_right + border

                # Calcul de la nouvelle largeur et hauteur
                width_new = X_right_new - X_left_new
                height_new = Y_right_new - Y_left_new
                width_current = X_right - X_left
                height_current = Y_right - Y_left

                # Ajuster le rapport d'aspect pour correspondre à 140:46
                current_aspect_ratio = width_new / height_new
                if current_aspect_ratio > self.target_aspect_ratio:
                    # Trop large : ajuster la hauteur
                    height_new = width_new / self.target_aspect_ratio
                else:
                    # Trop étroit : ajuster la largeur
                    width_new = height_new * self.target_aspect_ratio

                # Mise à jour des dimensions de la zone de recadrage
                if self.width_crop_max == 0 and self.height_crop_max == 0:
                    self.width_crop_max = width_new * 0.8  # Réduire la largeur de 20%
                    self.height_crop_max = height_new * 0.8  # Réduire la hauteur de 20%
                else:
                    self.width_crop_max += 1.5 * np.maximum(width_current - self.width_crop_max, 0)
                    self.height_crop_max += 1.5 * np.maximum(height_current - self.height_crop_max, 0)

                # Calcul des points de recadrage
                X_left_crop = int(X_center - self.width_crop_max / 2.0)
                X_right_crop = int(X_center + self.width_crop_max / 2.0)
                Y_left_crop = int(Y_center - self.height_crop_max / 2.0)
                Y_right_crop = int(Y_center + self.height_crop_max / 2.0)

                # __________ #
                ## _____ Ajoutez un filtre passe-bas pour lisser les coordonnées _____ #
                # class SmoothingFilter:
                #     def __init__(self, alpha=0.5):
                #         self.alpha = alpha
                #         self.prev_value = None

                #     def smooth(self, value):
                #         if self.prev_value is None:
                #             self.prev_value = value
                #         else:
                #             self.prev_value = self.alpha * value + (1 - self.alpha) * self.prev_value
                #         return self.prev_value

                # smoother = SmoothingFilter(alpha=0.5)
                # X_left_crop = smoother.smooth(X_left_crop)
                # X_right_crop = smoother.smooth(X_right_crop)
                # Y_left_crop = smoother.smooth(Y_left_crop)
                # Y_right_crop = smoother.smooth(Y_right_crop)

                # #______ Ajouter un décalagae _____ #
                # Décalage du rectangle
                decalage_x = int(self.width_crop_max * 0.06)   # vers la gauche
                decalage_y = int(self.height_crop_max * 0.065)  # vers le haut

                # Appliquer le décalage
                X_left_crop -= decalage_x
                X_right_crop -= decalage_x
                Y_left_crop -= decalage_y
                Y_right_crop -= decalage_y
                
                # __________ #

                # Vérification que les points de recadrage sont dans l'image
                if X_left_crop >= 0 and Y_left_crop >= 0 and X_right_crop < frame.shape[1] and Y_right_crop < frame.shape[0]:
                    mouth = frame[Y_left_crop:Y_right_crop, X_left_crop:X_right_crop, :]
                    activation = 1
                    lip_status = "Détecté"
                    lip_coordinates = (X_left_crop, Y_left_crop, X_right_crop - X_left_crop, Y_right_crop - Y_left_crop)  # (x, y, w, h)
                    cv2.rectangle(frame, (X_left_crop, Y_left_crop), (X_right_crop, Y_right_crop), (0, 255, 0), 2)
                else:
                    activation = 0
                    lip_status = "Hors champ"

        return frame, lip_coordinates, lip_status  # Retourner les coordonnées de la bouche