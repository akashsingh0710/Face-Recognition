#creating data sets
import cv2
import sys
import os
import shutil
import os.path
size = 4

fn_haar = r'C:\Apache24\htdocs\haarcascade_frontalface_default.xml'
fn_dir = r'C:\Apache24\htdocs\face_data' #All the faces data will be present this folder
fn_name = sys.argv[0]

path = r"C:\Apache24\htdocs\face_data\create_data"
print(path)
print(os.mkdir(path))

if os.path.exists(path):
    shutil.rmtree(path)
    print("a")
os.makedirs(path)

#if not os.path.isdir(path):
#    os.mkdir(path)
#    print('xt')
(im_width, im_height) = (240,240)
#(im_width, im_height) = (240,170)
haar_cascade = cv2.CascadeClassifier(fn_haar)
webcam = cv2.VideoCapture(0) #'0' is use for my webcam, if you've any other

 
# The program loops until it has 30 images of the face.
count = 0
while count < 60:
    (rval, im) = webcam.read()
    im = cv2.flip(im, 1, 0)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))
    faces = haar_cascade.detectMultiScale(mini)
    faces = sorted(faces, key=lambda x: x[3])
    if faces:
       face_i = faces[0]
       (x, y, w, h) = [v * size for v in face_i]
       face = gray[y:y + h, x:x + w]
       face_resize = cv2.resize(face, (im_width, im_height))
       pin=sorted([  int(n[:n.find('.')]) for n in os.listdir(path) if n[0]!='.' ]+[0])[-1] + 1
       print(pin)
       cv2.imwrite('%s/%s.jpg' % (path, pin), face_resize)
       cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
       cv2.putText(im, fn_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1,(0, 255, 0))
       count += 1
    
    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(5)
    if key == 27:
      break

