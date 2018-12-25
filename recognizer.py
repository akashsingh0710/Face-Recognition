#!/Python27/python

import cv2, numpy, os
fn_haar = 'C:\Apache24\htdocs\haarcascade_frontalface_default.xml'
fn_dir = 'C:\Apache24\htdocs\\face_data' #All the faces data will be present this folder

 
# Part 1: Training model
print('Training starts please wait....')
# Create a list of images and a list of corresponding names
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(fn_dir):
     for subdir in dirs:
         print(subdir)
         names[id] = subdir
         subjectpath = os.path.join(fn_dir, subdir)
         for filename in os.listdir(subjectpath):
              path = subjectpath + '/' + filename
              label = id
              images.append(cv2.imread(path, 0))
              labels.append(int(label))
         id += 1
(im_width, im_height) = (112, 92)
 
# Create array from the two lists above
(images, lables) = [ numpy.array(lis) for lis in [images, labels] ]
 
# OpenCV trains a model from the images
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, numpy.array(labels))
 
#Part 2: use LBPHFace recognizer on camera frame
face_cascade = cv2.CascadeClassifier(fn_haar)
webcam = cv2.VideoCapture(0)

while True:
      (_, im) = webcam.read()
      gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.3, 5)
      for (x,y,w,h) in faces:
           cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
           face = gray[y:y + h, x:x + w]
           face_resize = cv2.resize(face, (im_width, im_height))
           #Try to recognize the face
           prediction = model.predict(face_resize)
           cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
           #print &amp;quot;pred&amp;quot;,prediction
           if prediction[1]<100:
                 cv2.putText(im,'%s - %.0f' % (names[prediction[0]],
   prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
           else:
                 cv2.putText(im,'not recognized',(x-10, y-10),
                 cv2.FONT_HERSHEY_PLAIN,1,(0, 0,255))
 
      cv2.imshow('OpenCV', im)
      key = cv2.waitKey(10)
      if key == 27:
          break