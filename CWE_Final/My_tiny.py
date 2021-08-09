# install all required library which we will use further.
import cv2
import numpy as np
import time
import socket
import pandas as pd
import datetime

# Load your custome trained yolov4_tiny by passing weight and conf file
net = cv2.dnn.readNet("yolov4-tiny-custom_best.weights", "yolov4-tiny-custom.cfg")  # Original yolov3

classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# create a layer network layer from your yolov4_tiny model
layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# define colors to show it on bounding boxes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# create a empty csv files to store the predictions
columns = ['hostname', 'label', 'time', 'day']
emp_data = pd.DataFrame(columns=columns)

# get hostname your system to use as unique value in dataframe.
hostname_ = [socket.gethostname()]

# Take video using system's camera
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

# create a list to append the predictions
labels = []
times = []
days = []

# Most important to mentioned required time in seconds (60*5 = 300 - for 5 Minutes)
capture_duration = 60
start_time = time.time()

# this block of code will take frame from video and apply our model to each frame and will give predictions
# predictions will get appended into above define list and later on into csv file
# this will not show the video frame also, so employee wont get to know either model is model is analyzing his performance.
#

try:
    while (int(time.time() - start_time) < capture_duration):
        ret, frame = cap.read()
        if ret == True:

            ret, frame = cap.read()  #
            frame_id += 1

            height, width, channels = frame.shape
            # detecting objects
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)  # reduce 416 to 320

            net.setInput(blob)
            outs = net.forward(outputlayers)
            # print(outs[1])

            # Showing info on screen/ get confidence score of algorithm in detecting an object in blob
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3:
                        # onject detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                        # rectangle co-ordinaters
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                        boxes.append([x, y, w, h])  # put all rectangle areas
                        confidences.append(
                            float(confidence))  # how confidence was that object detected and show that percentage
                        class_ids.append(class_id)  # name of the object tha was detected

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, (255, 255, 255),
                                2)
                    labels.append(label)
                    times.append(datetime.datetime.now().strftime('%H:%M:%S %d/%m/%Y').split()[0])
                    days.append(datetime.datetime.now().strftime('%H:%M:%S %d/%m/%Y').split()[1])

                    print(label, datetime.datetime.now().strftime('%H:%M:%S %d/%m/%Y'))

            elapsed_time = time.time() - starting_time
            fps = frame_id / elapsed_time
            cv2.putText(frame, "FPS:" + str(round(fps, 2))
                        , (10, 50), font, 2, (0, 0, 0), 1)

            # cv2.imshow("Image", frame)
        else:
            break
except AttributeError:
    print('End of File Reached')
    cap.release()
    cv2.destroyAllWindows()
finally:
    ranges = range(0, len(labels), 10)

    emp_data['hostname'] = [sub.replace('-', ' ') for sub in hostname_] * len(labels)
    emp_data['label'] = labels
    emp_data['time'] = times
    emp_data['day'] = days
    #             emp_data.iloc[ranges].to_csv('results.csv',mode = 'a',index = None)
    p = []
    f_name = socket.gethostname()
    f_pth = 'C:/Users/Dell/Desktop/CWE_Final/Output/'

    file = f_pth + f_name + datetime.datetime.now().strftime('_%d-%m-%y_%H-%M')

    for i in range(0, len(labels), 10):
        p.append(emp_data.iloc[i:i + 10, :].mode().values[0])
        p_df = pd.DataFrame(p, columns=columns)

    p_df.to_csv(file + '.csv', mode='a', index=None)

cap.release() 
cv2.destroyAllWindows()
