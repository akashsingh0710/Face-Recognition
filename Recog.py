import numpy as np
import os, os.path
import cv2
import csv
import time
from numpy import linalg as LA
start_time = time.time()

imageDir = "C:\Apache24\htdocs\Database"
X = np.loadtxt('C:\\Apache24\\htdocs\\Optvec.txt')
pref=0
indn =0
for FP in sorted(os.listdir(imageDir)):
    num = 0
    image_list =sorted(os.listdir(imageDir+'\\'+ FP))
    for IP in image_list[8], image_list[9]:
        indn=indn+1
        print(indn)
        direc = imageDir+'\\'+ FP+'\\'+ IP
        Sm_img =  np.array(cv2.imread(direc,0))

        #Sm = cv2.cvtColor(cv2.imread(imageDir,0), cv2.COLOR_BGR2GRAY)
        fmatrix = np.transpose(np.dot(Sm_img,X))
        FSD  = fmatrix.reshape(560,1)
        Rname = []
        Fvecn = np.zeros((320,560))
        ind = 0
        with open(r'C:\Apache24\htdocs\fvector.csv') as f:
            reaer= csv.reader(f)
            for row in reaer:
                print(row)
                name = row[1]
                Fv = row[2:]
                Rname.append(name)
                col = 0
                for column in Fv:
#                    b_str= bytes(column, 'utf-8')
#                    my_str_as_bytes = column.encode()
#                    #my_decoded_str = my_str_as_bytes.decode()
#                    Fvecn[ind,col] = float(my_str_as_bytes.translate("'",None))
#                    intab = "[]'"
#                    outtab = None
#                    trantab = maketrans(intab, outtab)
                    
#                    Fvecn[ind,col] = column.translate(trantab)
                    Fvecn[ind,col] = column[1:-1]
                    #ggjg=" / ".join([item for sublist in column for item in sublist])
                    #Fvecn[ind,col] = [float(i) for i in column]
                    
                    #Fvecn[ind,col] = float(column[1:-1])
                    col = col + 1
                    continue
                ind = ind + 1
                continue

        Fvecdiff = np.zeros((320,1))
        r=0
        for row in Fvecn:
            Fvecdiff[r] = LA.norm(Fvecn[r,:]-np.transpose(FSD))
            r = r+1
            continue
        #print Fvecdiff
        #print np.argmin(Fvecdiff)
        #print Fvecdiff
        #print Rname[np.argmin(Fvecdiff)]
        if FP == Rname[np.argmin(Fvecdiff)]:
            pref = pref+1
        #print("--- %s seconds ---" % (time.time() - start_time))
print(pref)
