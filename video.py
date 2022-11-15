# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 01:03:06 2022

@author: PRAMILA
"""
from datetime import timedelta
import cv2
import numpy as np
import os,glob
from tensorflow.keras.preprocessing import image
SAVING_FRAMES_PER_SECOND = 5
import tensorflow as tf
import pathlib
model = tf.keras.models.load_model('model.h5')
import statistics
from statistics import mode

def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05) 
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")


def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

def prediction(video_file):
    print(video_file)
    filename, _ = os.path.splitext(video_file)
    
    filename += "-opencv"
    print(filename)
    # make a folder by the name of the video file
    if not os.path.isdir(filename):
        os.mkdir(filename)
    # read the video file    
    cap = cv2.VideoCapture(video_file)
    # get the FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    # get the list of duration spots to save
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    # start the loop
    count = 0
    while True:
        is_read, frame = cap.read()
        if not is_read:
            # break out of the loop if there are no frames to read
            break
        # get the duration by dividing the frame count by the FPS
        frame_duration = count / fps
        try:
            # get the earliest duration to save
            closest_duration = saving_frames_durations[0]
        except IndexError:
            # the list is empty, all duration frames were saved
            break
        if frame_duration >= closest_duration:
            # if closest duration is less than or equals the frame duration, 
            # then save the frame
            frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
            cv2.imwrite(os.path.join(filename, f"frame{frame_duration_formatted}.jpg"), frame) 
            # drop the duration spot from the list, since this duration spot is already saved
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass
        # increment the frame count
        count += 1


        # reading all the frames from temp folder
    # data_dir = pathlib.Path("D:/Phone/videos/")
    # images = list(glob(r"D:/Phone/videos/20210524_034033-opencv/*.jpg"))
    # data_dir = pathlib.Path(os.path.splitext(video_file))
    # print(data_dir)
    # directory="20210524_034033-opencv"
    # # filename1 = os.path.join(data_dir, directory)
    # # print(filename1)
    # images = (list(data_dir.glob('*/-opencv*/.jpg')))
    # print(images)
    # #images = r"D:/Phone/videos/"
    #print(len(images))
    prediction_images = []
    # for i in os.walk(filename+"/"):
    for root, dirs, files in os.walk(filename+"/"):
        sorted_files =  sorted(files)
        print(sorted_files)
        for file in sorted_files:
            if os.path.splitext(file)[1] == '.jpg':
                filePath = os.path.join(root, file)
   
                #prediction_images.append(img)
                img1 = image.load_img(filePath,target_size=(180,180))                        
                Y = image.img_to_array(img1)
                X = np.expand_dims(Y,axis=0)
                #val = model.predict(X)
                #imagea = np.vstack([X])

                r = model.predict(X)
                #print(r)
                val = int(np.argmax(r, axis=1))  
                print(val)
                #max_index_col = np.argmax(val, axis=1)    
                if val==1:
                    print("nofire")

                elif val == 0:
                    print("fire")
                prediction_images.append(val)
        return (mode(prediction_images))
        
        # # converting all the frames for a test video into numpy array
        # prediction_images = np.array(prediction_images)
        # # extracting features using pre-trained model
        # prediction_images = model.predict(prediction_images)
        # # converting features in one dimensional array
        # prediction_images = prediction_images.reshape(prediction_images.shape[0], 7*7*512)
        # # predicting tags for each array
        # prediction = model.predict_classes(prediction_images)
        # # appending the mode of predictions in predict list to assign the tag to the video
        # predict.append(y.columns.values[s.mode(prediction)[0][0]])
        # # appending the actual tag of the video
        # actual.append(videoFile.split('/')[1].split('_')[1])    
