from PIL import Image, ImageDraw
import face_recognition
import argparse


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename", required=True, help = "filename of image to \
detect faces")
args = vars(ap.parse_args())

# Load the jpg file into a numpy array
image = face_recognition.load_image_file(args["filename"])

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)

print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

for face_landmarks in face_landmarks_list:

    # Print the location of each facial feature in this image
    for facial_feature in face_landmarks.keys():
        print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))
        # print("The {} in this face has the following points: {}".format(facial_feat$)

    # Let's trace out each facial feature in the image with a line!
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)

    for facial_feature in face_landmarks.keys():
        d.line(face_landmarks[facial_feature], width=5)

    pil_image.show()


