
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image
import io
from datetime import timedelta
import os,glob
SAVING_FRAMES_PER_SECOND = 5
import cv2
from cv2 import *
from twilio.rest import Client
model = tf.keras.models.load_model('newmodel.h5')
import statistics
from statistics import mode
import shutil
import base64

st.set_page_config(
page_title="Forest Fire astrologer",
page_icon="🌳"
)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

def load_image(image_file):
	img = Image.open(image_file)
	if(image_file.type=="image/jpeg"):
	   picture = img.save("monuments.jpeg") 
	if(image_file.type=="image/jpg"):
	   picture = img.save("monuments.jpg") 
	return img



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
    if os.path.isdir(filename):
        shutil.rmtree(filename)
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
                img1 = image.load_img(filePath,target_size=(500,500))                        
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

add_bg_from_local("forestbg.jpg")
st.markdown("<h1 style ='color:black; text_align:center;font-family:times new roman;font-size:20pt; font-weight: bold;'>Forest Fire Identification from Satellite Images</h1>", unsafe_allow_html=True)
name=st.text_input("Please enter your name:")
phoneno=st.text_input("Please enter your phone no with prefix code (ex:+91):")
account_sid = "ACcebaf899ac73b0c205273410c2734d6b"
auth_token = "a8d7f33ada31dc69507fc86b724d47a3"
client = Client(account_sid, auth_token)

with st.expander("Image Based"):

	image_file = st.file_uploader("Upload Images", type=["jpeg","jpg"])


	if(st.button('Predict')):
		 # To See details
	    file_details = {"filename":image_file.name, "filetype":image_file.type,
			      "filesize":image_file.size}
	    st.write(file_details)

	    # To View Uploaded Image
	    st.image(load_image(image_file),width=200)

	    #image_file=load_image(image_file)

	  #  st.write(image_file.type)
	    categories = ['fire','nofire']

	    #model = tf.keras.models.load_model('model.h5')
	    #path = r"C:\Users\PRAMILA\.spyder-py3\project\monuments\ajantacaves\ajantacaves3.png"
	    if(image_file.type=="image/jpeg"):
	       img = image.load_img("monuments.jpeg", target_size=(500,500))
	    if(image_file.type=="image/jpg"):
	       img = image.load_img("monuments.jpg", target_size=(500,500))
	    #st.write(plt.imshow(img))
	    x = image.img_to_array(img)

	    x = np.expand_dims(x, axis=0)

	    #imagea = np.vstack([x])
	    r = model.predict(x)
	    #st.write(r)
	    # classes=round(classes[0][0])
	    # st.write(classes)

	    #type(classes)
	    classes = int(np.argmax(r, axis=1))   
	    #st.write(classes)
	    if classes==1:
	    	st.markdown(f"""<h1 style='text-align: center; font-weight:bold;color:black;background-color:#50C878;font-size:20pt;'>No Fire detected✅ </h1>""",unsafe_allow_html=True)
	    elif classes == 0:
	    	st.markdown(f"""<h1 style='text-align: center; font-weight:bold;color:black;background-color:#EE4B2B;font-size:20pt;'>Fire detected⚠️</h1>""",unsafe_allow_html=True)
	    	if(phoneno!=""):
	    		message = client.messages.create(body= 'Hi '+ name+'...ATTENTION! A fire has been detected in your area.',from_ =  "+15627845310",to = phoneno)
	    		print(message.sid)		

with st.expander("Video Based"):
	video_file = st.file_uploader("Upload video", type=["mp4"])


	if(st.button('Predict!')):
	    file_details = {"filename":video_file.name, "filetype":video_file.type,
			      "filesize":video_file.size}
	    st.write(file_details)
	    g = io.BytesIO(video_file.read())  ## BytesIO Object
	    temporary_location = "test.mp4"

	    with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
	    	out.write(g.read())  ## Read bytes into file

	    out.close()
	    video_file = open('test.mp4', 'rb')
	    video_bytes = video_file.read()
	    st.video(video_bytes)
	    with st.spinner('Please wait...'):
		    res=prediction("test.mp4")
		    if res==1:
		    	st.markdown(f"""<h1 style='text-align: center; font-weight:bold;color:black;background-color:#50C878;font-size:20pt;'>No Fire detected✅ </h1>""",unsafe_allow_html=True)
		    elif res == 0:
		    	st.markdown(f"""<h1 style='text-align: center; font-weight:bold;color:black;background-color:#EE4B2B;font-size:20pt;'>Fire detected⚠️</h1>""",unsafe_allow_html=True)
		    	if(phoneno!=""):
		    		message = client.messages.create(body= 'Hi '+ name+'...ATTENTION! A fire has been detected in your area.',from_ =  "+15627845310",to = phoneno)
		    		print(message.sid)			
