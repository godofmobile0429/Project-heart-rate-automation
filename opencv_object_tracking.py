# python opencv_object_tracking.py --video 1.mp4 --tracker csrt

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from cydets.algorithm import detect_cycles
import ctypes
import os

def Average(lst):
    return sum(lst) / len(lst)

def Mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)
def get_cycle(change_val):
    series = pd.Series(change_val)
    cycles = detect_cycles(series)
    arr = cycles['duration'];    
    
    max = arr[0]
    #Loop through the array    
    for i in range(0, len(arr)):    
        #Compare elements of array with max    
       if(arr[i] > max):    
           max = arr[i]
           
    return max
            
    

class Plotter:
    def __init__(self, plot_width, plot_height,sample_buffer=None):
        self.width = plot_width
        self.height = plot_height
        self.color = (255, 0 ,0)
        self.plot_canvas = np.ones((self.height, self.width, 3))*255    
        self.ltime = 0
        self.plots = {}
        self.plot_t_last = {}
        self.margin_l = 10
        self.margin_r = 10
        self.margin_u = 10
        self.margin_d = 50
        self.sample_buffer = self.width if sample_buffer is None else sample_buffer


    # Update new values in plot
    def plot(self, val, label = "plot"):
        if not label in self.plots:
            self.plots[label] = []
            self.plot_t_last[label] = 0
            
        self.plots[label].append(int(val))
        while len(self.plots[label]) > self.sample_buffer:
            self.plots[label].pop(0)
            self.show_plot(label)    
    # Show plot using opencv imshow
    def show_plot(self, label):

        self.plot_canvas = np.zeros((self.height, self.width, 3))*255
        cv2.line(self.plot_canvas, (self.margin_l, int((self.height-self.margin_d-self.margin_u)/2)+self.margin_u ), (self.width-self.margin_r, int((self.height-self.margin_d-self.margin_u)/2)+self.margin_u), (0,0,255), 1)        

        # Scaling the graph in y within buffer
        scale_h_max = max(self.plots[label])
        scale_h_min = min(self.plots[label]) 
        scale_h_min = -scale_h_min if scale_h_min<0 else scale_h_min
        scale_h = scale_h_max if scale_h_max > scale_h_min else scale_h_min
        scale_h = ((self.height-self.margin_d-self.margin_u)/2)/scale_h if not scale_h == 0 else 0
        

        for j,i in enumerate(np.linspace(0,self.sample_buffer-2,self.width-self.margin_l-self.margin_r)):
            i = int(i)
            cv2.line(self.plot_canvas, (j+self.margin_l, int((self.height-self.margin_d-self.margin_u)/2 +self.margin_u- self.plots[label][i]*scale_h)), (j+self.margin_l, int((self.height-self.margin_d-self.margin_u)/2  +self.margin_u- self.plots[label][i+1]*scale_h)), self.color, 1)
        
        
        cv2.rectangle(self.plot_canvas, (self.margin_l,self.margin_u), (self.width-self.margin_r,self.height-self.margin_d), (255,255,255), 1) 
        cv2.putText(self.plot_canvas,f" {label} : {self.plots[label][-1]} , dt : {int((time.time() - self.plot_t_last[label])*1000)}ms",(int(0),self.height-20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        cv2.circle(self.plot_canvas, (self.width-self.margin_r, int(self.margin_u + (self.height-self.margin_d-self.margin_u)/2 - self.plots[label][-1]*scale_h)), 2, (0,200,200), -1)
        
        self.plot_t_last[label] = time.time()
        cv2.imshow(label, self.plot_canvas)
        cv2.waitKey(1)


def get_similarity(image1, image2):
    """This function returns the absolute
    value of the entered number"""
    nsumFrameColor = 0.0
    
    gray_image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray_image], [0], 
                         None, [256], [0, 256])
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    histogram2 = cv2.calcHist([gray_image2], [0], 
                          None, [256], [0, 256])
    i = 0
    while i<len(histogram) and i<len(histogram2):
        nsumFrameColor+=(histogram[i]-histogram2[i])**2
        i+= 1
    nsumFrameColor = nsumFrameColor**(1 / 2)
    
    return nsumFrameColor



def get_frame_value(image, width, height):
    """This function returns the absolute
    value of the entered number"""
    nsumFrameColor = 0.0
    for i in range(width):
        for j in range(height):
            color = image[j,i]
            nsumFrameColor += float(color[0]/(width*height))
   
    return nsumFrameColor

def plotScatter(plt, x, y):
    plt.scatter(x, y);
    plt.pause(0.05)
    


xlist = list()
ylist = list()
frameno = 0;

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
                help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
                help="OpenCV object tracker type")
args = vars(ap.parse_args())

# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]

# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_create(args["tracker"].upper())

# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "mil": cv2.TrackerMIL_create,

    }

    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# initialize the FPS throughput estimator
fps = None
fps_video = vs.get(cv2.CAP_PROP_FPS)

total_frame_count = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
file_len_sec = float(total_frame_count/fps_video)

index = 0
v = 0
# loop over frames from the video stream
p = Plotter(400, 200,sample_buffer = 200)

first_Image = None
change_val = list()
cycle_value = None
cycle_value_second = list()
cycles_frame = fps_video * 4

while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    # check to see if we have reached the end of the stream
    if frame is None:
        break

    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    frame = imutils.resize(frame, width=1000)
    (H, W) = frame.shape[:2]

    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = (1, initBB)

        # check to see if the tracking was a success
        if success:
            
            
          #plotScatter(plt,index,int(math.sin(index*3.14/180)*100));
              
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            crop_img = frame[y:y + h, x:x + w]
            if(int(index) == 0):
                first_Image = crop_img
                
            fileName = "cropped_" + str(index) + "_.jpg"
            index += 1
            cv2.imshow("cropped field", crop_img)
            cl = get_similarity(first_Image,crop_img)
            change_val.append(cl)
            length = len(change_val)

            if(length ==  cycles_frame):
                cycle_value = get_cycle(change_val)
                
                change_val = list()
                length = 0
                if(cycle_value != None):
                    
                    heartrate_sec =float( cycle_value / fps_video)
                    heartrate_min = int(60/heartrate_sec)
                    cycle_value_second.append(heartrate_min)
            
            p.plot(cl,label='Real-time Result')
            
            #cv2.imwrite(fileName, crop_img)
        area = (w - x) * (h - y)

        #print(x, y, w, h)
        #print(area)
        
        frameno += 1;
        xlist.append(frameno);
        ylist.append(area);
        # plt.plot(xlist,ylist,'-o');
        # plt.show()
        # plt.pause(0.0001)
        # update the FPS counter

        fps.update()

        fps.stop()

        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),

        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                               showCrosshair=False)
        #print(initBB)
        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)
        fps = FPS().start()
        
        
    # cv2.imshow('image',initBB)


    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()

# otherwise, release the file pointer
else:
    vs.release()


optimize_size = len(cycle_value_second)
opt_list = list()
opt_list = cycle_value_second[1:]

space_time = float(file_len_sec/(optimize_size - 1))

x = np.arange(0, file_len_sec, float(file_len_sec/ len(opt_list) ))
#print(len(x))
time_list = list()

for i in range(len(opt_list)):
    val_insert = str(int(i * space_time )) + " s" + " - " + str(int((i+1) * space_time )) + " s"
    time_list.append(val_insert)


fig = plt.figure()
# Instead of set_figwidth(30)
fig.set_size_inches(fig.get_figwidth(), 2, forward=True)

plt.title("BPS graph")
plt.xlabel("Time Line (s)")
plt.ylabel("B P S")
#plt.plot(x, opt_list)
#print(len(opt_list))
#plt.show()

average_rate = Average(opt_list)

data_save = {'From-To': time_list, 'Heart rate': opt_list, 'Average heart rate(/min)': average_rate}
df_save = pd.DataFrame(data = data_save)
df_save.to_csv('result_bps.csv')

Mbox("Success message", "Result saved to 'result_bps.csv' successfully!", 0)
#os.system("Excel.exe" + 'result_bps.csv')
# close all windows
#cv2.destroyAllWindows()