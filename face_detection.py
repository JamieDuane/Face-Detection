import PIL.Image as pi
import PIL.ImageDraw as pid
import face_recognition as fr

class FaceDetection(object):
    def __init__(self, image=None):
        self.image = image
        if image is not list:
            self.face_locations = fr.face_locations(image)
            self.face_encodings = fr.face_encodings(image)
            self.face_landmarks_list = fr.face_landmarks(image)
            self.pil_image = pi.fromarray(self.image)
            self.draw = pid.Draw(self.image)

    def find_number_of_faces(self, show_info=False):
        number_of_faces = len(self.face_locations)
        if show_info:
            print(f"I found {number_of_faces} face(s) in this photograph.")
        return number_of_faces

    def draw_face_location(self, show_info=False, show_faces=True):
        for face_location in self.face_locations:
            top, right, bottom, left = face_location
            if show_info:
                print(f"A face is located at pixel location Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}")
            self.draw.rectangle([left, top, right, bottom], outline="red")
        if show_faces:
            self.pil_image.show()

    def draw_face_landmark(self, show_info=False, show_faces=True):
        for face_landmarks in self.face_landmarks_list:
            for name, list_of_points in face_landmarks.items():
                if show_info:
                    print(f"The {name} in this face has the following points: {list_of_points}")
                self.draw.line(list_of_points, fill="red", width=2)
        if show_faces:
            self.pil_image.show()

    def get_info_of_image_list(self):
        if type(self.image) is list:
            face_locations_list, face_encodings_list, face_landmarks_list = [], [], []
            for img in self.image:
                face_locations_list.append(fr.face_locations(img))
                face_encodings_list.append(fr.face_encodings(img))
                face_landmarks_list.append(fr.face_landmarks(img))
            return face_locations_list, face_encodings_list, face_landmarks_list
        return

    def make_up_digitally(self, show_image=True):
        d = pid.Draw(self.pil_image, 'RGBA')
        for face_landmarks in self.face_landmarks_list:
            # The face landmark detection model returns these features:
            #  - chin, left_eyebrow, right_eyebrow, nose_bridge, nose_tip, left_eye, right_eye, top_lip, bottom_lip

            # Draw a line over the eyebrows
            d.line(face_landmarks["left_eyebrow"], fill=(128, 0, 128, 100), width=3)
            d.line(face_landmarks["right_eyebrow"], fill=(128, 0, 128, 100), width=3)

            # Draw over the lips
            d.polygon(face_landmarks['top_lip'], fill=(128, 0, 128, 100))
            d.polygon(face_landmarks['bottom_lip'], fill=(128, 0, 128, 100))
        if show_image:
            self.pil_image.show()