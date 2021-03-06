import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from sklearn.cluster import KMeans

import cv2
import os 
import numpy
import os
from gtts import gTTS
from googletrans import Translator
from playsound import playsound
from pydub.playback import play

#load the trained model to classify sign
from keras.models import load_model
import random
model = load_model('C:/Users/Shreevara/Desktop/capstone/datatsetssss/3/sequential.h5')

#dictionary to label all traffic signs class.
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }
                 
#initialise GUI
traffic=tk.Tk()
traffic.geometry('700x600')
traffic.title('Recognize Traffic Sign')
traffic.configure(background="#E67E22")



label=Label(traffic,background='#E67E22', font=('Comic Sans MS',20,'italic'))
sign_image = Label(traffic)
s_image = Label(traffic)



def classify(file_path):
    i=0
    image = Image.open(file_path)
    image = image.resize((32,32))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    print(image.shape)
    #pred = model.predict([image])[0]
    pred=numpy.argmax(model.predict(image))
    sign = classes[pred]
    print(sign)
    label.configure(foreground='#011638', text=sign) 
    TTS = gTTS(text=sign, lang="en-IN")
    x=random.randint(0,100)
    #TTS = gTTS(text=Text, lang="en-IN")
    d=str(os.getcwd())+"/"+"voice.mp3"
    TTS.save(d)
    playsound(d)
    os.remove(d)
    
   
def detect(file_path):
    #x = Image.open(file_path)
    #print(file_path)
    img =  cv2.imread(os.path.join(file_path))
    #img = numpy.array(img)
    #img="C://Users//saran//OneDrive//Documents//CAPSTONE//images.jfif"
    #img=cv2.imread(img)
    #img = numpy.array(img)
    
    
    
    if img.shape[0]>img.shape[1]:
        img=cv2.resize(img,(300,350))
    else:
        img=cv2.resize(img,(350,300))
        
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_op=img 
    #masking the red regions
    mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
    mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))
    
    #masking the blue regions
    #mask3=cv2.inRange(img_hsv, (100,150,20), (130,255,255))
    
    ## Merge the mask and crop the regions
    mask = cv2.bitwise_or(mask1, mask2)
    croped = cv2.bitwise_and(img, img, mask=mask)
    
    img=croped
    imgContour=img.copy()
        
    imgBlur=cv2.GaussianBlur(img,(7,7),1)
    imgGray=cv2.cvtColor(imgBlur,cv2.COLOR_BGR2GRAY)
    
    #cv2.imshow("Result",img)
    #cv2.imshow("Blur",imgBlur)
    #cv2.imshow("Gray",imgGray)
    
    imgCanny=cv2.Canny(imgGray,80,80)
    kernel=numpy.ones((5,5))
    imgDil=cv2.dilate(imgCanny,kernel,iterations=1)
    getContours(imgDil,imgContour,img_op)
    

       
   
     
    
    
    
    
    
def getContours(img,imgContour,img_op):
    contours,hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    i=0
    img1=img_op 
    x=os.getcwd()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>900:
            cv2.drawContours(imgContour,cnt,-1,(255,0,255),7)
            peri = cv2.arcLength(cnt,True)
            approx= cv2.approxPolyDP(cnt,0.02*peri,True)
            x,y,w,h=cv2.boundingRect(approx)
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),5)
            new_img=img1[y:y+h,x:x+w]
            #save the cropped photo
            cv2.imwrite(str(x)+str(i)+'.png',new_img)
            y=str(x)+str(i)+'.png'
            print(x)
            i+=1
            
            uploaded=Image.open(y)
            uploaded.thumbnail(((traffic.winfo_width()/2.25),(traffic.winfo_height()/2.25)))
            im=ImageTk.PhotoImage(uploaded)
            sign_image.configure(image=im)
            sign_image.image=im
    classify(y)
         
            

            
            
            

def getNoise(img,imgContour):
    noise=[]
    contours,hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area<1000:
            noise.append(cnt)
            cv2.drawContours(imgContour,cnt,-1,(255,0,255),7)
            peri = cv2.arcLength(cnt,True)
            approx= cv2.approxPolyDP(cnt,0.02*peri,True)
            x,y,w,h=cv2.boundingRect(approx)
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,0,0),5)
    return noise

def crop(img,co_ord,out):
    for i in co_ord:
        x,y,w,h=i[0],i[1],i[2],i[3]
        new_img=img[y:y+h, x:x+w]
        out.append(new_img)

dict={1:'???',2:'???',3:'???',4:'???',5:'???',6:'???',7:'???',8:'??????',9:'???',10:'???',11:'???',12:'???',13:'???',14:'???',15:'??????',16:'??????',17:'???',18:'??????',19:'??????',20:'??????',21:'??????',22:'??????',23:'??????',24:'??????',25:'??????',26:'??????',27:'??????',28:'??????',29:'???',30:'??????',31:'??????',32:'??????',33:'???',34:'??????',35:'??????',36:'??????',37:'??????',38:'??????',39:'??????',40:'??????',41:'??????',42:'??????',43:'???',44:'??????',45:'???',46:'??????',47:'???',48:'??????',49:'??????',50:'??????',51:'??????',52:'???',53:'???',54:'??????',55:'??????',56:'??????',57:'??????',58:'??????',59:'??????',60:'??????',61:'???',62:'??????',63:'??????',64:'??????',65:'??????',66:'??????',67:'??????',68:'??????', 69:'???',70:'??????',71:'??????',72:'??????',73:'??????',74:'??????',75:'??????',76:'???',77:'??????',78:'??????',79:'???',80:'??????',81:'??????',82:'???',83:'??????',84:'??????',85:'??????',86:'??????',87:'????????????',88:'??????',89:'?????????',90:'???',91:'??????',92:'??????',93:'??????',94:'???',95:'??????',96:'??????',97:'??????',98:'??????',99:'??????',100:'??????',101:'??????',102:'??????',103:'??????',104:'??????',105:'???',106:'??????',107:'??????',108:'??????',109:'???',110:'??????',111:'??????',112:'??????',113:'??????',114:'??????',115:'??????',116:'??????',117:'??????',118:'??????',119:'??????',120:'??????',121:'??????',122:'???',123:'??????',124:'??????',125:'??????',126:'??????',127:'??????',128:'??????',129:'??????',130:'??????',131:'??????',132:'??????',133:'???',134:'??????',135:'??????',136:'??????',137:'??????',138:'??????',139:'???',140:'??????',141:'??????',142:'??????',143:'??????',144:'??????',145:'??????',146:'??????',147:'??????',148:'??????',149:'???',150:'??????',151:'??????',152:'??????',153:'??????',154:'??????',155:'??????',156:'??????',157:'??????',158:'???',159:'??????',160:'??????',161:'??????',162:'??????',163:'??????',164:'??????',165:'??????',166:'??????',167:'??????',168:'??????',169:'???',170:'??????',171:'??????',172:'??????',173:'??????',174:'??????',175:'??????',176:'???',177:'??????',178:'??????',179:'??????',180:'??????',181:'??????',182:'??????',183:'??????',184:'??????',185:'??????',186:'??????',187:'??????',188:'??????',189:'???',190:'??????',191:'??????',192:'??????',193:'??????',194:'??????',195:'??????',196:'??????',197:'??????',198:'??????',199:'???',200:'??????',201:'??????',202:'??????',203:'??????',204:'??????',205:'??????',206:'??????',207:'???',208:'??????',209:'??????',210:'??????',211:'??????',212:'??????',213:'??????',214:'??????',215:'???',216:'??????',217:'??????',218:'??????',219:'??????',220:'??????',221:'??????',222:'???',223:'??????',224:'??????',225:'??????',226:'???',227:'??????',228:'??????',229:'??????',230:'??????',231:'??????',232:'??????',233:'??????',234:'??????',235:'??????',236:'??????',237:'???',238:'??????',239:'???',240:'????????????'}

from tensorflow.keras.models import load_model
import tensorflow as tf

def preprocess(image):
    img_gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) #converting to grayscale
    #otsu's method of binarization
    thresh,img_bin=cv2.threshold(img_gray,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img_bin

def consec_zero(hist_row):
    res=[]
    count=0
    prev=hist_row[0]
    for i in range(len(hist_row)):
        if hist_row[i]==0:
            count+=1
            prev=0
        else:
            if prev==0:
                res.append(count)
                count=0
            prev=hist_row[i]
    res.append(count) 
    return res

def trans(file_path):
    global label_packed
    #img=Image.open(file_path)
    img =  cv2.imread(os.path.join(file_path))
    
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #converting to RGB format
    img_col=img.copy()
    
    img_bin=preprocess(img)
    img_canny=cv2.Canny(img_bin,90,90)
    kernel=numpy.ones((5,5))
    img_dil=cv2.dilate(img_canny,kernel,iterations=1)
    
    img_contour=img.copy()
    noise=getNoise(img_dil,img_contour)
    
    #filling the noise region with black pixels
    cv2.drawContours(img_bin, noise, -1, color=(0,0,0), thickness=cv2.FILLED)
    img_hor=numpy.array(img_bin)
    
    #histogram of horizontal sum of pixels 
    hist_col = numpy.sum(img_bin, axis = 1)
    
    #drawing horizontal lines were the sum of pixels is less than the threshold
    thresh = 1
    H,W = img_hor.shape[:2]
    uppers = [y for y in range(H-1) if hist_col[y]<=thresh and hist_col[y+1]>thresh]
    lowers = [y for y in range(H-1) if hist_col[y]>thresh and hist_col[y+1]<=thresh]

    for y in uppers:
        cv2.line(img_hor, (0,y), (W, y), (255,255,255), thickness=1)

    for y in lowers:
        cv2.line(img_hor, (0,y), (W, y), (255,255,255), thickness=1)
    
    #getting the cordinates of the ROI
    line_wise=[]
    for i in range(len(uppers)):
        temp=[0,uppers[i],W,lowers[i]-uppers[i]]
        line_wise.append(temp)
        
    #line wise segmentation
    line_seg_img=[]
    line_seg_col_img=[]
    crop(img_bin,line_wise,line_seg_img)
    crop(img_col,line_wise,line_seg_col_img)
    
    temp_col=[]
    temp_bin=[]
    for j in range(len(line_seg_img)):
        hist_row = numpy.sum(line_seg_img[j], axis = 0)
        H,W = line_seg_img[j].shape
        uppers=[]
        for i in range(len(hist_row)):
            if hist_row[i+60]>1:
                uppers.append(i)
                break

        for i in range(len(hist_row)-1,-1,-1):
            if hist_row[i-60]>1:
                uppers.append(i)
                break

        word_wise=[]

        temp=[uppers[0],0,uppers[1]-uppers[0],H] #x,y,w,h
        word_wise.append(temp)
 
        temp=[]
        crop(line_seg_col_img[j],word_wise,temp)
        temp_col.append(temp[0])
    
        temp=[]
        crop(line_seg_img[j],word_wise,temp)
        temp_bin.append(temp[0])

    line_seg_col_img=temp_col
    line_seg_img=temp_bin
    
    word_seg_img=[]
    word_seg_col_img=[]
    out=[]
    for i in range(len(line_seg_img)):
        hist_row = numpy.sum(line_seg_img[i], axis = 0)
        out+=consec_zero(hist_row)
    
        #drawing vertical lines were the sum of pixels is less than the threshold
        H,W = line_seg_img[i].shape
        th=1
        th1=H//3
        uppers = [y for y in range(W-1) if hist_row[y]<=th and hist_row[y+1]>th]
        lowers = [y for y in range(W-1) if hist_row[y]>th and hist_row[y+1]<=th]

        for y in uppers:
            cv2.line(line_seg_img[i], (y,0), (y,H), (255,255,255), 2)

        for y in lowers:
            cv2.line(line_seg_img[i], (y,0), (y,H), (255,255,255), 2)
    
        #getting the cordinates of the ROI
        word_wise=[]
        for k in range(len(uppers)):
            temp=[uppers[k],0,lowers[k]-uppers[k],H] #x,y,w,h
            word_wise.append(temp)
    
        temp=[]
        crop(line_seg_col_img[i],word_wise,temp)
        word_seg_col_img.append(temp)
        
    #k-means clustering
    X=[]
    for i in range(len(out)):
        X.append([0,out[i]])
    X=numpy.array(X) 
  
    #creating the an object of sklearn.cluster.KMeans class    
    kmeans=KMeans(n_clusters=2)    
  
    #passing the data to the model  
    kmeans.fit(X)  
   
    #Printing the predicted values
    res=kmeans.labels_
    
    new_model=load_model('C:/Users/Shreevara/Desktop/Capstone/translation/model_seq1.h5')
    

    out=[]
    for i in word_seg_col_img:
        for j in i:
            img = cv2.resize(j,(32,32))
            pred=numpy.argmax(new_model.predict(img[None]), axis=-1)
            out.append(dict[pred[0]+1]) 
        out.append(" ")
    out.pop()

    res=res.tolist()
    res.pop()
    res.pop(0)
    if res[1]==1:
        target=0
    else:
        target=1
    
    j=1
    text=""+out[0]

    for i in range(len(res)):
        if out[j]!=" ":
            if res[i]==target:
                text+=" "
            text+=out[j]
        j+=1
        
    out=text
    print(out)
    
    translator = Translator()
    #print(type(out))
    out1 = translator.translate(out,dest="en")
    Text = out1.text
    print(Text)
    #print(out)
    label.configure(foreground='#011638', text=Text)
    TTS = gTTS(text=Text, lang="en-IN")
    print(os.getcwd())
    dev=str(os.getcwd())+"\\voice.mp3"
    TTS.save(dev)
    playsound(dev)
    os.remove(dev)





def show_recognize_button(file_path):
    detect_b=Button(traffic,text="Translate Sign",command=lambda: trans(file_path),padx=5,pady=5)
    detect_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx=0.79,rely=0.66)
    
def show_detect_button(file_path):
    print(file_path)
    detect_b=Button(traffic,text="Detect Sign",command=lambda: detect(file_path),padx=15,pady=5)
    detect_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx=0.79,rely=0.56)
     
    

def show_classify_button(file_path):
    classify_b=Button(traffic,text="Recognize Sign",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('Candara',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((traffic.winfo_width()/2.25),(traffic.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image=im)
        sign_image.image=im
        
        
        
        label.configure(text='')
        show_classify_button(file_path)
        show_recognize_button(file_path)
        show_detect_button(file_path)
        
        
        
    except:
        pass






upload=Button(traffic,text="Upload a traffic sign",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('Candara',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)



heading = Label(traffic, text="Know The Traffic Sign",pady=20, font=('Gabriola',25,'bold'))
heading.configure(background='#FBE8C0',foreground='#2A064F')


heading.pack()




traffic.mainloop()
