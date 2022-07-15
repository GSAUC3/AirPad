import handMediapipe as hm 
import cv2,numpy as np


def convolution(image):
    kernel = np.zeros((160,160),dtype=np.uint8)
    kernel[2:-2,2:-2]=1
    ans = (kernel*image).sum()/25218
    np.set_printoptions(threshold=np.inf)
    if ans>10:
        return True
    return False


def draw_grid(image,size=80*2):
    '''
    This function will draw a grid on the output window
    '''
    for i in range(1,4):
        cv2.line(image,(i*size,0),(i*size,480-160),(255,255,255),1)
    for i in range(1,3):
        cv2.line(image,(0,i*size),(640,i*size),(255,255,255),1)
    return image

cap = cv2.VideoCapture(0)
writingpad = np.zeros((480,640,3),np.uint8)
xprev,yprev =0,0
var = hm.handTrack(1)
while 1:
    # i=0
    _, frame = cap.read()
    frame = cv2.flip(frame,1)

    a,landmarks,image = var.fingersUD(frame)
    
    if landmarks:
        _,x,y,_ = landmarks['index'][-1] # returns the coordinate of the tip of the index finger  
        if a['index']==1 and a['middle']==0 and a['ring']==0:
            
            if xprev==0 and yprev==0:
                xprev,yprev = x,y 
            cv2.line(image,(xprev,yprev),(x,y),(255,255,255),8)
            cv2.line(writingpad,(xprev,yprev),(x,y),(255,255,255),8)
            xprev,yprev = x,y 

        elif a['index']==1 and a['middle']==1 and a['ring']==0:
            xprev,yprev =0,0
        elif a['index']==1 and a['middle']==1 and a['ring']==1:

            pad = cv2.cvtColor(writingpad,cv2.COLOR_BGR2GRAY)
 
            
        if a['index']==1 and a['middle']==1 and a['ring']==1 and a['pinky']==1 and a['thumb']==1:
            writingpad[:,:,:] = 0
         

   


    # i+=1
    image = cv2.addWeighted(image,0.5,writingpad,0.6,0.0)
    # merging the two images (i.e., writing pad and the actual image)
    image= draw_grid(image)
    cv2.imshow('output',cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    # cv2.imshow('output2',writingpad)
    if cv2.waitKey(1)== ord('s'):
        cv2.imwrite('applePie.jpg',writingpad)
        
    elif cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()