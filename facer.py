#!/Python27/python
import numpy as np
import os, os.path
import cv2
import csv
from numpy import linalg as LA


imageDir = 'C:\Apache24\htdocs\Database\\'

#Calculating average image of all samles
Avg_img = np.zeros((112,92))
# number of samples
M = 0         
for FP in sorted(os.listdir(imageDir)):            #FP means Folder Path
    num = 0
    for IP in sorted(os.listdir(imageDir+FP)):          #IP means Image Path
        img = np.array(cv2.imread(imageDir+FP+'\\'+IP,0))
        Avg_img = Avg_img + img
        if num== 8:
           break
        num=num+1
        M = M+1
        continue
    continue
print('m =' , M)
Avg_img = Avg_img/M
# Calculating image covariance matrix
G_l = np.zeros((92,92))
for FP in sorted(os.listdir(imageDir)):
    num = 0
    for IP in sorted(os.listdir(imageDir+FP)) :
        img = np.array(cv2.imread(imageDir+FP+'\\'+IP,0))
        meandiff = img-Avg_img
        G_l = G_l + np.dot(np.transpose(meandiff), meandiff)
        if num==8:
           break
        num=num+1
        continue
    continue
G_l = G_l/M
print(M,num)
w, v = LA.eig((G_l))
wn = np.array(w)
vn = np.array(v)
opv = vn[:,:5]

np.savetxt('Optvec.txt',opv)
zp =0
for FP in sorted(os.listdir(imageDir)):
    num = 0
    
    for IP in sorted(os.listdir(imageDir+FP)) :
        img = np.array(cv2.imread(imageDir+FP+'\\'+IP,0))
        fmatrix = np.transpose(np.dot(img,opv))
        FSD  = fmatrix.reshape(560,1).tolist()
        l_vector = []
        l_vector = list([zp])
        if num==8:
           break
        num=num+1
        zp=zp+1
        l_vector.append(FP)
        l_vector.extend(FSD)
        with open('fvector.csv', 'a', newline='') as csvfile:
             writer = csv.writer(csvfile )          #, quoting=csv.QUOTE_NONE
             writer.writerow(l_vector)
        continue
    continue
print(sorted(os.listdir(imageDir+FP)))
#    img = np.array(cv2.imread(imageDir+FP+'/'+IP,0))
#    fmatrix = np.transpose(np.array([ np.dot(img,opv[:,0]), np.dot(img,opv[:,1]), np.dot(img,opv[:,2]), np.dot(img,opv[:,3]), np.dot(img,opv[:,4])]))
#    FSD  = fmatrix.reshape(560,1).tolist()
    #l_vector = []
#    l_vector = list([FP])
    
    #l_vector = l_vector.append(FP)
#    l_vector.append(FSD)
#    with open('fvector.csv', 'a') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(l_vector)
#   print len(FSD)
    #print FP
    #f_handle = file('Fvector.txt','a')
    #np.savetxt(f_handle, fmatrix)
    #f_handle.close()
#    continue
#imageDir = '/home/Akash/Desktop/Database/s10/8.pgm'
#testimg = np.array(cv2.imread(imageDir, 0))




















#print fmresult[:,:,:]
#cv2.imshow('A_img',A_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#w, v = LA.eig((G_l))
#wn = np.array(w)
#vn = np.array(v)
# Taking only 5 eigenvectors corresponds to 5 largest eigenvalues
#opv = vn[:,:5]
# Calculating feature matrix
#fmatrix = np.transpose(np.array([ np.dot(img,opv[:,0]), np.dot(img,opv[:,1]), np.dot(img,opv[:,2]), np.dot(img,opv[:,3]), np.dot(img,opv[:,4])]))
#height, width = fvector.shape[:2]
#print fmatrix[:,1]
#print num
#f_handle = file('Fvector.txt','a')
#np.savetxt(f_handle, fmresult)
#f_handle.close()


