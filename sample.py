import cv2
import mediapipe as mp
import numpy as np
import vg
import math
import copy
import matplotlib.pyplot as plt
import time
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

landmark_to_joint_angles_0=[]
landmark_to_joint_angles_1=[]
# For webcam input use 0, else try 1:
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('cyberglove.mov')

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        if results.multi_handedness[idx].classification[0].label=="Left":
          landmark_to_joint_angles_0.append(np.insert(calc_joint_angles(hand_landmarks).T.flatten(), 0, time.time()))
        elif results.multi_handedness[idx].classification[0].label=="Right":
          landmark_to_joint_angles_1.append(np.insert(calc_joint_angles(hand_landmarks).T.flatten(), 0, time.time()))

        ### if you want to visualize the joint angles, you can uncomment this
        mp_drawing.draw_landmarks(
          image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
          
        #joints=calc_joint_angles(hand_landmarks)
        # print(joints)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    #cv2.imshow('MediaPipe Hands', image)

    #press esc to exit
    if cv2.waitKey(5) & 0xFF == 27:
      break

print(landmark_to_joint_angles_0)
print(landmark_to_joint_angles_1)

np.savetxt("C:/Users/mxw00/Documents/meng/hand-interaction-capture/sample_data/landmark_to_joint_angles_left.csv", landmark_to_joint_angles_0, delimiter=",")
np.savetxt("C:/Users/mxw00/Documents/meng/hand-interaction-capture/sample_data/landmark_to_joint_angles_right.csv", landmark_to_joint_angles_1, delimiter=",")

cap.release()

