from ttkbootstrap import *
import cv2
from tkinter import *
from PIL import Image,ImageTk
import handMediapipe as hm 
import numpy as np
import Model
from easyocr import Reader
from datetime import datetime
from tkinter import messagebox

class window:
    def __init__(self,root):
        self.root=root
        self.root.title('Air Pad')
        self.root.resizable(0,0)
        
        self.root.iconbitmap('icon/icon.ico')
        self.var = hm.handTrack(1)
        self.writingpad = np.zeros((480,640,3),np.uint8)
        self.xprev,self.yprev =0,0

        self.root.bind("<Escape>",lambda x:self.root.quit()) 
        self.root.bind("<s>",lambda x:self.predicshun()) 
        self.root.bind("<Control-s>",lambda x:self.__savetext()) 
        self.root.bind("<Control-e>",lambda x:self.__saveimg()) 

        self.now = datetime.now()

        menu= Menu(self.root)
        self.file = Menu(menu)
        self.file.add_command(label='8 Letters Mode',command=self.__draw_grid)
        self.file.add_command(label='Single Letter Mode',command=self.__nogrid)
        self.file.add_command(label='Save Text',command=self.__savetext)
        self.file.add_command(label='Save Full Image',command=self.__saveimg)
        self.file.add_separator()
        self.file.add_command(label='Exit', command=self.root.quit)
        menu.add_cascade(label='File', menu=self.file)
        self.root.config(menu=menu)

        self.grid=False

        self.l1=ttk.Label(self.root)
        self.l1.pack()
        self.cap=cv2.VideoCapture(0)
        ttk.Button(self.root,text='Predict',width=15,command=self.predicshun).pack(side='right')
        self.update()

        self.l2=ttk.Label(self.root)
        self.l2.pack()

        self.reader = Reader(['en'])
        self.text = ''

    def __saveimg(self):
        date_time = self.now.strftime("%H%M%S-%d-%m-%Y")
        cv2.imwrite(f'{date_time}.png',self.writingpad)
        messagebox.showinfo('Image Saved',f"Image saved as {date_time}.png")
        

    def __savetext(self):
        if self.text == '':
            messagebox.showwarning('Empty Text',"You haven't written anything!!!")
        else:
            date_time = self.now.strftime("%H%M%S-%d-%m-%Y")
            with open(f'{date_time}.txt','w') as f:
                f.write('bal')
            
            messagebox.showinfo('Text Saved',f"Text saved as {date_time}.txt")

    def __draw_grid(self):
        self.grid=True 

    def __nogrid(self):
        self.grid=False

    def draw_grid(self,image,size=80*2):
        '''
        This function will draw a grid on the output window
        '''
        for i in range(1,4):
            cv2.line(image,(i*size,0),(i*size,480-160),(255,255,255),1)
        for i in range(1,3):
            cv2.line(image,(0,i*size),(640,i*size),(255,255,255),1)
        return image

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
        
            
    def update(self):
        ret,frame=self.cap.read()
        # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame,1)
        a,landmarks,image = self.var.fingersUD(frame)

        if ret:
 
            if landmarks:
                _,x,y,_ = landmarks['index'][-1] # returns the coordinate of the tip of the index finger  
                if a['index']==1 and a['middle']==0 and a['ring']==0:
                    
                    if self.xprev==0 and self.yprev==0:
                        self.xprev,self.yprev = x,y 
                    
                    cv2.line(self.writingpad,(self.xprev,self.yprev),(x,y),(255,255,255),8)
                    self.xprev,self.yprev = x,y 

                elif a['index']==1 and a['middle']==1 and a['ring']==0:
                    self.xprev,self.yprev =0,0

                if a['index']==1 and a['middle']==1 and a['ring']==1 and a['pinky']==1 and a['thumb']==1:
                    self.writingpad[:,:,:] = 0
                

            image = cv2.addWeighted(image,0.5,self.writingpad,0.6,0.0)
            if self.grid:
                image= self.draw_grid(image)

            self.pic=ImageTk.PhotoImage(image=Image.fromarray(image))
            self.l1['image']=self.pic
        
        self.root.after(10,self.update)

    def predicshun(self):
        if self.grid:
            letters = Model.sliding_window(self.writingpad)
            self.l2.config(text=''.join(letters))
        else:
            letters = self.reader.readtext(self.writingpad,detail=0)
            self.l2.config(text=' '.join(letters))
            self.text += ' '.join(letters)

if __name__ == '__main__':
    win=Style(theme='darkly').master
    app=window(win)
    
    win.mainloop()