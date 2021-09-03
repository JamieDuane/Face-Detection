from face_detection import *
import face_recognition as fr
from pathlib import Path

class FaceRecognition(FaceDetection):
    def __init__(self, known_image_name=None, unknown_image=None):
        self.known_image_name = known_image_name
        if type(known_image_name) is str:
            known_image = fr.load_image_file(known_image_name)
        elif type(known_image_name) is list:
            known_image = [fr.load_image_file(k) for k in known_image_name]
        else:
            known_image = None
        self.unknown_image = unknown_image
        super().__init__(image=known_image)

    def get_info_of_unknown_image(self):
        unknown_face_locations, unknown_face_landmarks= fr.face_locations(self.unknown_image,number_of_times_to_upsample=2), fr.face_landmarks(self.unknown_image)
        unknown_face_encodings = fr.face_encodings(self.unknown_image, known_face_locations=unknown_face_locations)
        return unknown_face_locations, unknown_face_encodings, unknown_face_landmarks

    def recognize_face(self, show_info=True):
        unknown_face_locations, unknown_face_encodings, unknown_face_landmarks = self.get_info_of_image_list()

        if type(self.image) is list:
            known_face_locations_list, known_face_encodings_list, known_face_landmarks_list = self.get_info_of_image_list()
        elif self.image is None:
            return
        else:
            known_face_encodings_list = [self.face_encodings]

        results_name = []
        for unknown_face_encoding in unknown_face_encodings:
            results = fr.compare_faces(known_face_encodings_list, unknown_face_encoding, tolerance=0.4)
            for i in range(len(results)):
                if results[i]:
                    results_name.append(self.known_image_name[i])
                    if show_info:
                        print(f"Found {self.known_image_name[i]} in the photo!")
        return results_name

    def get_similarity(self):
        if type(self.image) is not list and type(self.image) is not None:
            best_face_distance = 1.0
            best_face_image = None
            known_image_encoding = self.face_encodings[0]
            directory = ''
            for image_path in Path(directory).glob("*.png"):
                unknown_image = fr.load_image_file(image_path)
                face_encodings = fr.face_encodings(unknown_image)
                face_distance = fr.face_distance(face_encodings, known_image_encoding)
                if face_distance < best_face_distance:
                    best_face_distance = face_distance
                    best_face_image = unknown_image
            pil_image = pi.fromarray(best_face_image)
            pil_image.show()

