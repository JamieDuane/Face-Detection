from face_recognition_system import FaceRecognition, FaceDetection
from pathlib import Path
import face_recognition, os

def get_input_image(image_name):
    path = os.path.curdir
    image_folder = path + '/data/images'
    files = Path(image_folder).glob('*.jpg')
    path = Path('.')/'data'
    image = face_recognition.load_image_file(path/image_name)
    return image, files

def get_output(image=None, imagenamelist=None, mode='detection', function='noofface'):
    if mode == 'detection' and image is not None:
        if function == 'noofface':
            fd = FaceDetection(image)
            fd.find_number_of_faces(show_info=True)
        elif function == 'location':
            fd = FaceDetection(image)
            fd.draw_face_location(show_info=True)
        elif function == 'landmarks':
            fd = FaceDetection(image)
            fd.draw_face_landmark(show_info=True)
        elif function == 'makeup':
            fd = FaceDetection(image)
            fd.make_up_digitally()
    elif mode == 'recognition' and image is not None and imagenamelist is not None:
        if function == 'normal':
            fr = FaceRecognition(known_image_name=imagenamelist, unknown_image=image)
            fr.recognize_face(show_info=True)
        elif function == 'similarity':
            fr = FaceRecognition(known_image_name=imagenamelist, unknown_image=image)
            fr.get_similarity()

def main(imagename):
    image, imagenamelist = get_input_image(imagename)
    get_output(image=image, imagenamelist=imagenamelist, mode='recognition', function='normal')

main('unknown_2.jpg')
