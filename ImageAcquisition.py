import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_alt2.xml')
face_cascade1 = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_alt.xml')
face_cascade2 = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_alt_tree.xml')
face_cascade3 = cv2.CascadeClassifier('Cascades/data/lbpcascade_frontalface_improved.xml')
face_cascade5 = cv2.CascadeClassifier('Cascades/data/lbpcascade_frontalface.xml')
#face_cascade6 = cv2.HOGDescriptor('Cascades/data/hogcascade_pedestrians.xml')
#face_cascade6.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())



face_cascade_profile = cv2.CascadeClassifier('Cascades/data/haarcascade_profileface.xml')
face_cascade4 = cv2.CascadeClassifier('Cascades/data/lbpcascade_profileface.xml')


#cap = cv2.VideoCapture(0)
entireFrame = cv2.imread("Images/Sample.jpeg")
stroke = 2
newx,newy = 1280,720

newframe = cv2.resize(entireFrame,(newx,newy))
x1,y1 = 0,0
x2,y2 = 200,200
frame=0
while True:
    while True:
        frame = newframe[y1:y2,x1:x2]
        #ret, frame = cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.5, minNeighbors = 3, flags=0, minSize = (0,0), maxSize = (300,300))
        for(x,y,w,h) in faces:
            print(x,y,w,h)
            roi_gray = gray[y:y+h,x:x+w]
            roi_color = frame[y:y+h,x:x+w]
            
            
            img_item = "my-image.jpeg"
            cv2.imwrite(img_item, roi_gray)
            
            color = (255,0,0)
            
            endcord_x = x + w
            endcord_y = y + h
            cv2.rectangle(frame,(x,y),(endcord_x,endcord_y),color,stroke)
            
        faces1 = face_cascade1.detectMultiScale(gray,scaleFactor = 1.5, minNeighbors = 3, flags=0, minSize = (0,0), maxSize = (100,100))
        for(x,y,w,h) in faces1:
            endcord_x = x + w
            endcord_y = y + h
            cv2.rectangle(frame,(x,y),(endcord_x,endcord_y),(0,255,0),stroke)
        
        faces2 = face_cascade2.detectMultiScale(gray,scaleFactor = 1.5, minNeighbors = 3, flags=0, minSize = (0,0), maxSize = (100,100))
        for(x,y,w,h) in faces2:
            endcord_x = x + w
            endcord_y = y + h
            cv2.rectangle(frame,(x,y),(endcord_x,endcord_y),(0,0,255),stroke)
            
        faces3 = face_cascade3.detectMultiScale(gray,scaleFactor = 1.5, minNeighbors = 3, flags=0, minSize = (0,0), maxSize = (100,100))
        for(x,y,w,h) in faces3:
            endcord_x = x + w
            endcord_y = y + h
            cv2.rectangle(frame,(x,y),(endcord_x,endcord_y),(255,255,0),stroke)
        
          
        faces4 = face_cascade4.detectMultiScale(gray,scaleFactor = 1.5, minNeighbors = 3, flags=0, minSize = (0,0), maxSize = (100,100))
        for(x,y,w,h) in faces4:
            endcord_x = x + w
            endcord_y = y + h
            cv2.rectangle(frame,(x,y),(endcord_x,endcord_y),(255,0,255),stroke)
            
           
        faces5 = face_cascade5.detectMultiScale(gray,scaleFactor = 1.5, minNeighbors = 3, flags=0, minSize = (0,0), maxSize = (100,100))
        for(x,y,w,h) in faces5:
            endcord_x = x + w
            endcord_y = y + h
            cv2.rectangle(frame,(x,y),(endcord_x,endcord_y),(255,255,255),stroke)
        
            
        '''faces6 = face_cascade6.detectMultiScale(gray,winStride = (8,8), padding = (16,16), scale=1.5, useMeanShiftGrouping = 1)
        for(x,y,w,h) in faces6:
            endcord_x = x + w
            endcord_y = y + h
            cv2.rectangle(frame,(x,y),(endcord_x,endcord_y),(0,64,255),stroke)
        '''    
            
        profile_faces = face_cascade_profile.detectMultiScale(gray,scaleFactor = 1.5, minNeighbors = 3, flags=0, minSize = (0,0), maxSize = (100,100))
        for(x,y,w,h) in profile_faces:
            endcord_x = x + w
            endcord_y = y + h
            cv2.rectangle(frame,(x,y),(endcord_x,endcord_y),(0,255,255),stroke)
        if(x2<newx):
            print("2nd if Entered!")
            x1+=40
            x2+=40
        elif((x2>=newx) and (y2<newy)):
            print("3rd if Entered!")
            x1=0
            x2=200
            y1+=52
            y2+=52
        else:
            print("Breaking Loop")
            break
        cv2.imshow('frame',frame)
        c = cv2.waitKey(0)
        if(ord('q')==c):
            print("Next!")    
    
           
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    
    
cv2.destroyAllWindows()
