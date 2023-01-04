import cv2
import threading
import mediapipe as mp
import numpy as np
import vg

import time

from scipy.spatial.transform import Rotation as R
import pandas as pd
import os


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

new_mp_drawing = mp.solutions.drawing_utils

file_path = "C:/Users/mxw00/Documents/meng/hand-interaction-capture"

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    def run(self):
        print("Starting " + self.previewName)
        cam_preview(self.previewName, self.camID)


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

def get_palm_area(hand_landmarks):
    palm_pts=[0, 5, 9, 13, 17]
    x=[]
    y=[]
    for i in palm_pts:
        x+=[hand_landmarks.landmark[mp_hands.HandLandmark(i)].x]
        y+=[hand_landmarks.landmark[mp_hands.HandLandmark(i)].y]
    return 0.5*np.abs(np.dot(np.array(x),np.roll(np.array(y),1))-np.dot(np.array(y),np.roll(np.array(x),1)))


 



##############################################################


def cam_preview(previewName, camID):
    landmark_to_joint_angles_right=[]
    landmark_to_joint_angles_left=[]

    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
            if cam.isOpened():
                rval, frame = cam.read()
            else:
                rval = False

            while rval:
                rval, frame = cam.read()
                key = cv2.waitKey(20)
    
                frame.flags.writeable = False
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        if get_palm_area(hand_landmarks)>=0.01:
                            if idx==0:
                                landmark_to_joint_angles_right.append(np.insert(calc_joint_angles(hand_landmarks).T.flatten(), 0, [time.time(), results.multi_handedness[idx].classification[0].score, get_palm_area(hand_landmarks)]))
                            elif idx==1:
                                landmark_to_joint_angles_left.append(np.insert(calc_joint_angles(hand_landmarks).T.flatten(), 0, [time.time(), results.multi_handedness[idx].classification[0].score, get_palm_area(hand_landmarks)]))
                        
                        ### if you want to visualize the joint angles, you can uncomment this
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow(previewName, cv2.flip(image, 1))
                if key == 27:  # exit on ESC
                    break
    cv2.destroyWindow(previewName)

    np.savetxt(file_path+"/left/landmark_to_joint_angles_left_"+str(camID)+".csv", landmark_to_joint_angles_left, delimiter=",")
    np.savetxt(file_path+"/right/landmark_to_joint_angles_right_"+str(camID)+".csv", landmark_to_joint_angles_right, delimiter=",")

    

def start_cam_threads(numCams):
    if not os.path.exists(file_path+"/left"):
        os.makedirs(file_path+"/left")
    if not os.path.exists(file_path+"/right"):
        os.makedirs(file_path+"/right")
    threads = [camThread("Camera "+str(i+1), i) for i in range(numCams)]
    [t.start() for t in threads]
    print("Active threads", threading.activeCount()) 
    [t.join() for t in threads]
    consolidate_hands()

def consolidate_hands():
    file_list_left = os.listdir(file_path+"/left")
    file_list_right = os.listdir(file_path+"/right")
    df_concat_left = pd.DataFrame()
    df_concat_right = pd.DataFrame()
    for f in file_list_left:
        if os.path.getsize(file_path+"/left/"+f)>0:
            df_concat_left = pd.concat([df_concat_left, pd.read_csv(file_path+"/left/"+f, index_col=False)], ignore_index=True)
    
    for f in file_list_right:
        if os.path.getsize(file_path+"/right/"+f)>0:
            df_concat_right = pd.concat([df_concat_right, pd.read_csv(file_path+"/right/"+f, index_col=False)], ignore_index=True)

    if not df_concat_left.empty:
        df_concat_left.sort_values(df_concat_left.columns[0], 
                        axis=0,
                        inplace=True)
    if not df_concat_right.empty:
        df_concat_right.sort_values(df_concat_right.columns[0], 
                        axis=0,
                        inplace=True)
        
    df_concat_left.to_csv(file_path+'/joint_angles_left.csv', index=False)
    df_concat_right.to_csv(file_path+'/joint_angles_right.csv', index=False)
   
start_cam_threads(1)





