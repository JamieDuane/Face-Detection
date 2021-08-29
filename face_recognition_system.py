import face_recognition

# Load the known images
image1 = face_recognition.load_image_file('person_1.jpg')
image2 = face_recognition.load_image_file('person_2.jpg')
image3 = face_recognition.load_image_file('person_3.jpg')

# Get the face encoding of each person. This can fail if no one is found in the photo.
face_encodings1 = face_recognition.face_encodings(image1)[0]
face_encodings2 = face_recognition.face_encodings(image2)[0]
face_encodings3 = face_recognition.face_encodings(image3)[0]

# Create a list of all known face encodings
known_face_encodings = [
    face_encodings1,
    face_encodings2,
    face_encodings3
]

# Load the image we want to check
image_unknown = face_recognition.load_image_file('unknown_7.jpg')

# Get face encodings for any people in the picture
face_locations = face_recognition.face_locations(image_unknown, number_of_times_to_upsample=2)
face_encodings_unknown = face_recognition.face_encodings(image_unknown, known_face_locations=face_locations)

# There might be more than one person in the photo, so we need to loop over each face we found
for unknown_face_encoding in face_encodings_unknown:

    # Test if this unknown face encoding matches any of the three people we know
    results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)

    name = "Unknown"

    if results[0]:
        name = "Person 1"
    elif results[1]:
        name = "Person 2"
    elif results[2]:
        name = "Person 3"

    print(f"Found {name} in the photo!")
