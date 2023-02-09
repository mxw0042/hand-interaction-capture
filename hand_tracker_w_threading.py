import cv2
import mediapipe as mp
import numpy as np
import vg
import math
import copy
import matplotlib.pyplot as plt
import time
import pyaudio
import wave
import os
import _thread

from threading import Thread
from mediapipe.framework.formats import landmark_pb2
from scipy.spatial.transform import Rotation as R

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

new_mp_drawing = mp.solutions.drawing_utils

### extract joint angles from positions

#get normal vector of the plane formed by 3 points
def get_palm_normal(landmark1, landmark2, landmark3):
  #points
  p1=np.array([landmark1.x, landmark1.y, landmark1.z])
  p2=np.array([landmark2.x, landmark2.y, landmark2.z])
  p3=np.array([landmark3.x, landmark3.y, landmark3.z])

  #vectors
  v1 = p2 - p1
  v2 = p3 - p1
  
  normal=np.cross(v1, v2)
  return normal

#get normal vector of the plane formed by 3 points
def get_cross(landmark0, landmark1, normal):
  p0=np.array([landmark0.x, landmark0.y, landmark0.z])
  p1=np.array([landmark1.x, landmark1.y, landmark1.z])

  segment_direction = p1 - p0
  cross=np.cross(normal, segment_direction)
  return cross

#base flexion/adduction
def calc_alpha_beta(landmark0, landmark1, landmark2, normal):
  p0=np.array([landmark0.x, landmark0.y, landmark0.z])
  p1=np.array([landmark1.x, landmark1.y, landmark1.z])
  p2=np.array([landmark2.x, landmark2.y, landmark2.z])

  v1 = p0 - p1
  v2 = p2 - p1

  cross=np.cross(-v1, normal)
  alpha= np.pi - vg.angle(v1, v2, look=cross, units='rad')
  beta= vg.signed_angle(v2, v1, look=normal, units='rad')-np.pi

  return alpha, beta

#flexion
def calc_PIP_DIP(landmark1, landmark2, landmark3, landmark4, normal):
  p1=np.array([landmark1.x, landmark1.y, landmark1.z])
  p2=np.array([landmark2.x, landmark2.y, landmark2.z])
  p3=np.array([landmark3.x, landmark3.y, landmark3.z])
  p4=np.array([landmark4.x, landmark4.y, landmark4.z])

  #vectors
  v1 = p1 - p2
  v2 = p2 - p3
  v3 = p3 - p4

  cross=np.cross(-v1, normal)

  angle_PIP= vg.angle(v1, v2, look=cross, units='rad')
  angle_DIP = vg.angle(v2, v3, look=cross, units='rad')

  return angle_PIP, angle_DIP

# go from 21 hand landmarks embedded in landmarks object to array of joint angles
def calc_joint_angles(hand_landmarks):
  joint_angles=[]

  landmarks=[]
  for i in range(21):
    landmarks.append(hand_landmarks.landmark[mp_hands.HandLandmark(i)])

  thumb_index_normal=get_palm_normal(landmarks[1], landmarks[2], landmarks[5])
  index_middle_normal=get_palm_normal(landmarks[0], landmarks[5], landmarks[9])
  middle_ring_normal=get_palm_normal(landmarks[0], landmarks[9], landmarks[13])
  ring_pinky_normal=get_palm_normal(landmarks[0], landmarks[13], landmarks[17])

  normals=[thumb_index_normal, index_middle_normal, middle_ring_normal, middle_ring_normal, ring_pinky_normal]

  for i in range(5):
    alpha, beta=calc_alpha_beta(landmarks[0], landmarks[1+4*i], landmarks[2+4*i], normals[i])
    pip, dip=calc_PIP_DIP(landmarks[1+4*i], landmarks[2+4*i], landmarks[3+4*i], landmarks[4+4*i], normals[i])
    
    joint_angles+=[alpha, pip, dip, beta]

  return np.array([joint_angles]).T

##############################################################

file_path="C:/Users/mxw00/Documents/meng/federico" #"C:/Users/chopp/Dropbox (MIT)/PianoProject/MediaPipeHandTracking"

class VideoWriter(object):
    def __init__(self, video_file_name, src=0):
        # Create a VideoCapture object
        self.frame_name = str(src) # if using webcams, else just use src as it is.
        self.video_file = video_file_name
        self.video_file_name = video_file_name + '.avi'
        self.capture = cv2.VideoCapture(src)

        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))

        # Set up codec and output video settings
        self.codec = cv2.VideoWriter_fourcc('M','J','P','G')
        self.output_video = cv2.VideoWriter(file_path+"/"+self.video_file_name, self.codec, 30, (self.frame_width, self.frame_height))

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        self.landmark_to_joint_angles_0=[]
        self.landmark_to_joint_angles_1=[]

        open(file_path + "/right/landmark_to_joint_angles_RIGHT_"+self.video_file+".csv", 'w').close()
        open(file_path + "/left/landmark_to_joint_angles_LEFT_"+self.video_file+".csv", 'w').close()

        # Start another thread to show/save frames
        self.start_recording()
        print('initialized {}'.format(self.video_file))

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def show_frame(self):
        # Display frames in main program
         with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            if self.status:
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                self.frame.flags.writeable = False
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                results = hands.process(self.frame)

                # Draw the hand annotations on the image.
                self.frame.flags.writeable = True
                #self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        handedness = results.multi_handedness[idx].classification[0].label
                        if handedness=='Left':
                            self.landmark_to_joint_angles_0 = np.insert(calc_joint_angles(hand_landmarks).T.flatten(), 0, time.time())
                        elif handedness=='Right':
                            self.landmark_to_joint_angles_1 = np.insert(calc_joint_angles(hand_landmarks).T.flatten(), 0, time.time())

                        ### if you want to visualize the joint angles, you can uncomment this
                        mp_drawing.draw_landmarks(
                            self.frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                cv2.imshow(self.frame_name, cv2.flip(self.frame, 1))

            # Press esc on keyboard to stop recording
            if cv2.waitKey(5) & 0xFF == 27:
                self.capture.release()
                self.output_video.release()
                cv2.destroyAllWindows()
                
                _thread.interrupt_main()
                exit(1)

    def save_frame(self):
        # Save obtained frame into video output file
        self.output_video.write(cv2.flip(self.frame, 1))
        #print(self.landmark_to_joint_angles_0)
        with open(file_path + "/right/landmark_to_joint_angles_RIGHT_"+self.video_file+".csv", "a") as f:
            np.savetxt(f, self.landmark_to_joint_angles_0.reshape(1, -1), delimiter=",")
        with open(file_path + "/left/landmark_to_joint_angles_LEFT_"+self.video_file+".csv", "a") as f:
            np.savetxt(f, self.landmark_to_joint_angles_1.reshape(1, -1), delimiter=",")
        # np.savetxt(file_path + "/right/landmark_to_joint_angles_RIGHT_"+self.video_file+".csv", self.landmark_to_joint_angles_0, delimiter=",")
        # np.savetxt(file_path + "/left/landmark_to_joint_angles_LEFT_"+self.video_file+".csv", self.landmark_to_joint_angles_1, delimiter=",")
        self.landmark_to_joint_angles_0=[]
        self.landmark_to_joint_angles_1=[]
    def start_recording(self):
        # Create another thread to show/save frames
        def start_recording_thread():
            while True:
                try:
                    self.show_frame()
                    self.save_frame()
                except AttributeError:
                    pass
        self.recording_thread = Thread(target=start_recording_thread, args=())
        self.recording_thread.daemon = True
        self.recording_thread.start()
        

if __name__ == '__main__':
    if not os.path.exists(file_path+"/left"):
        os.makedirs(file_path+"/left")
    if not os.path.exists(file_path+"/right"):
        os.makedirs(file_path+"/right")
    src1 = 0 #default webcam
    video_writer1 = VideoWriter('Camera 1', src1)
    src2 = 1 #external webcam
    video_writer2 = VideoWriter('Camera 2', src2)

    # Since each video player is in its own thread, we need to keep the main thread alive.
    # Keep spinning using time.sleep() so the background threads keep running
    # Threads are set to daemon=True so they will automatically die 
    # when the main thread dies
    while True:
        time.sleep(1)