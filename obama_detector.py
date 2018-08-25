import dlib
from scipy.spatial.distance import euclidean
import pickle

# Set the model paths
shape_predictor_path = 'shape_predictor_5_face_landmarks.dat'
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'

obama_vectors = pickle.load( open( "obama.p", "rb" ) )
# Set the detector and shape predictor functions
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(shape_predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# Load the image you want to recognize
img = dlib.load_rgb_image('obama.jpg')

dets = detector(img, 1)

for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        #print(face_descriptor)
        distance = euclidean(face_descriptor, obama_vectors)
        print('Distance between pre-computed facial vectors and picture we give is: ' +str(distance))