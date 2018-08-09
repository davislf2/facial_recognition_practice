# facial_recognition_practice



# Face Recognition Tutorial

**June-16-2018**

**Speaker : Blue**



## Environment

### Install MiniConda

### Create Environment

```
conda create --name py35 python=3.5 
```

You will have a conda environment named “py35”.

Start the environment:

```
source activate py35
```

## Install required face recognition modules

### 1. Install Dlib:

[install dlib on different platforms](https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/)

### Example: Install dlib on macos

Install homebrew

```
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew update
```

Set homebrew environment

```
echo -e "\n# Homebrew" >> ~/.bash_profile
echo "export PATH=/usr/local/bin:$PATH" >> ~/.bash_profile
source ~/.bash_profile
```

Install CMake to compile dlib

```
brew install cmake
```

Then we install other required python libraries

```
pip install numpy
pip install jupyter
pip install scikit-image
pip install dlib
pip install imutils
pip install scikit-learn
pip install cython
```

Check if your dlib is properly installed

```
(py35) Fu-Chuns-MacBook-Pro:tutorial_June16 bluekidds$ ipython
Python 3.5.5 |Anaconda, Inc.| (default, Apr 26 2018, 08:11:22) 
Type 'copyright', 'credits' or 'license' for more information
IPython 6.4.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import dlib

In [2]: dlib.__version__
Out[2]: '19.13.1'
```

### 2. Test Your Dlib

```
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

    # Let's trace out each facial feature in the image with a line!
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)

    for facial_feature in face_landmarks.keys():
        d.line(face_landmarks[facial_feature], width=5)

    pil_image.show()
```

Copy the file and save to test_dlib.py, download any single face image and save it as example.jpg(or any name you like)

```
python test_dlib.py -f example.jpg
```

### Install OpenCV (Optional)

If you want to play deeper into face applications and see the result through visualization, you will need to install opencv3. However, this will take quite long to install(>0.5hr).

```
conda install -c menpo opencv3 
```

### 3. Install facial_recognition

```
pip install face_recognition
### Play with the code
```

## Examples

### 1-1. Face Detector(HOG)

```
from PIL import Image
import dlib
import time

# Add this only to compute the time spent.
start_time = time.time()

detector = dlib.get_frontal_face_detector()

# Load the jpg file into a numpy array

image = dlib.load_rgb_image('example.jpg')
print('Size of your image: ' + str(image.shape))
detector = dlib.get_frontal_face_detector()

dets = detector(image,1)
for i, d in enumerate(dets):

    # You can access the actual face itself like this:
        face_image = image[d.top():d.bottom(), d.left():d.right()]
        pil_image = Image.fromarray(face_image)
        pil_image.show()

print("--- %s seconds ---" % (time.time() - start_time))
```

### 1-2. Face Detector(CNN)

Basically similar, except now we use different detector. Simply replace the following:

\#Note: Download the model from:[model](http://dlib.net/files/mmod_human_face_detector.dat.bz2)

```
detector dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
```

### 2. Face Landmark Detection

Download the landmark detection model from

```
 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```

It is also a very large file, you can try this at home.
In the folloing we will give an example of how to detect face landmarks:

```
# import the necessary packages
from imutils import face_utils
import dlib
import cv2

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# load the input image and convert it to grayscale
image = cv2.imread("example.jpg")

# Resize the image to 1/2 size of original
small = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 0)

# loop over the face detections
for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

# show the output image with the face detections + facial landmarks
cv2.imshow("Output", small)
cv2.waitKey(0)
```

### 3. Facial Recognition - web service

```
import face_recognition
from flask import Flask, jsonify, request, redirect

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # The image file seems valid! Detect faces and return the result.
            return detect_faces_in_image(file)

    # If no valid image file was uploaded, show the file upload form:
    return '''
    <!doctype html>
    <title>Is this a picture of Obama?</title>
    <h1>Upload a picture and see if it's a picture of Obama!</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''


def detect_faces_in_image(file_stream):
    # Pre-calculated face encoding of Obama generated with face_recognition.face_encodings(img)
    known_face_encoding = [-0.09634063,  0.12095481, -0.00436332, -0.07643753,  0.0080383,
                            0.01902981, -0.07184699, -0.09383309,  0.18518871, -0.09588896,
                            0.23951106,  0.0986533 , -0.22114635, -0.1363683 ,  0.04405268,
                            0.11574756, -0.19899382, -0.09597053, -0.11969153, -0.12277931,
                            0.03416885, -0.00267565,  0.09203379,  0.04713435, -0.12731361,
                           -0.35371891, -0.0503444 , -0.17841317, -0.00310897, -0.09844551,
                           -0.06910533, -0.00503746, -0.18466514, -0.09851682,  0.02903969,
                           -0.02174894,  0.02261871,  0.0032102 ,  0.20312519,  0.02999607,
                           -0.11646006,  0.09432904,  0.02774341,  0.22102901,  0.26725179,
                            0.06896867, -0.00490024, -0.09441824,  0.11115381, -0.22592428,
                            0.06230862,  0.16559327,  0.06232892,  0.03458837,  0.09459756,
                           -0.18777156,  0.00654241,  0.08582542, -0.13578284,  0.0150229 ,
                            0.00670836, -0.08195844, -0.04346499,  0.03347827,  0.20310158,
                            0.09987706, -0.12370517, -0.06683611,  0.12704916, -0.02160804,
                            0.00984683,  0.00766284, -0.18980607, -0.19641446, -0.22800779,
                            0.09010898,  0.39178532,  0.18818057, -0.20875394,  0.03097027,
                           -0.21300618,  0.02532415,  0.07938635,  0.01000703, -0.07719778,
                           -0.12651891, -0.04318593,  0.06219772,  0.09163868,  0.05039065,
                           -0.04922386,  0.21839413, -0.02394437,  0.06173781,  0.0292527 ,
                            0.06160797, -0.15553983, -0.02440624, -0.17509389, -0.0630486 ,
                            0.01428208, -0.03637431,  0.03971229,  0.13983178, -0.23006812,
                            0.04999552,  0.0108454 , -0.03970895,  0.02501768,  0.08157793,
                           -0.03224047, -0.04502571,  0.0556995 , -0.24374914,  0.25514284,
                            0.24795187,  0.04060191,  0.17597422,  0.07966681,  0.01920104,
                           -0.01194376, -0.02300822, -0.17204897, -0.0596558 ,  0.05307484,
                            0.07417042,  0.07126575,  0.00209804]

    # Load the uploaded image file
    img = face_recognition.load_image_file(file_stream)
    # Get face encodings for any faces in the uploaded image
    unknown_face_encodings = face_recognition.face_encodings(img)

    face_found = False
    is_obama = False

    if len(unknown_face_encodings) > 0:
        face_found = True
        # See if the first face in the uploaded image matches the known face of Obama
        match_results = face_recognition.compare_faces([known_face_encoding], unknown_face_encodings[0])
        if match_results[0]:
            is_obama = True

    # Return the result as json
    result = {
        "face_found_in_image": face_found,
        "is_picture_of_obama": is_obama
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
```

### 4. Facial Recognition - Compute your facial feature

Download your triplet-loss model at [face recognition](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2) and [5-point landmark](http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2)

We will use the example of the previous computed Obama’s feature comparing with himself. It **should be less than 0.6**

```
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
```