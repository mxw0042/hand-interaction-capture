import cv2
import mediapipe as mp
import numpy as np
import vg
import math
import copy
import matplotlib.pyplot as plt
from mediapipe.framework.formats import landmark_pb2
from scipy.spatial.transform import Rotation as R

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

new_mp_drawing = mp.solutions.drawing_utils

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

def normalize(arr):
  normalized_vector = arr / np.linalg.norm(arr)
  return normalized_vector

#reconstruct position from flexion and abduction angles
def reconstruct_pip(alpha, beta, normal, landmark0, landmark1, landmark2, handedness):
  p0=np.array([landmark0.x, landmark0.y, landmark0.z])
  p1=np.array([landmark1.x, landmark1.y, landmark1.z])
  p2=np.array([landmark2.x, landmark2.y, landmark2.z])

  segment_magnitude=math.dist(p1, p2)
  segment_direction = p1 - p0
  segment= segment_magnitude*normalize(segment_direction)

  cross=np.cross(normal, segment_direction)

  r_alpha = R.from_rotvec(alpha * normalize(cross))
  if handedness=="Right":
    r_beta = R.from_rotvec(beta * normalize(-normal))
  else: 
    r_beta = R.from_rotvec(beta * normalize(normal))
  rotated_vector_beta = r_beta.apply(segment)
  rotated_vector_alpha = r_alpha.apply(rotated_vector_beta)


  landmark_array=p1+rotated_vector_alpha 
  landmark=copy.copy(landmark2)
  landmark.x=landmark_array[0]
  landmark.y=landmark_array[1]
  landmark.z=landmark_array[2]

  return landmark

#reconstruct joint position from flexion angle
def reconstruct_dip(alpha, normal, landmark0, landmark1, landmark2, base, cross):
  p0=np.array([landmark0.x, landmark0.y, landmark0.z])
  p1=np.array([landmark1.x, landmark1.y, landmark1.z])
  p2=np.array([landmark2.x, landmark2.y, landmark2.z])
  base=np.array([base.x, base.y, base.z])

  segment_magnitude=math.dist(p1, p2)
  segment_direction = base - p0
  segment= segment_magnitude*normalize(segment_direction)

  # print("alpha: ", alpha)

  r_alpha = R.from_rotvec(alpha * normalize(cross))

  rotated_vector_alpha = r_alpha.apply(segment)


  landmark_array=base+rotated_vector_alpha 
  landmark=copy.copy(landmark2)
  
  landmark.x=landmark_array[0]
  landmark.y=landmark_array[1]
  landmark.z=landmark_array[2]

  #build landmark object?
  return landmark

#reconstruct finger tip position from flexion angle
def reconstruct_tip(alpha, normal, base0, landmark1, landmark2, base1, cross):
  p1=np.array([landmark1.x, landmark1.y, landmark1.z])
  p2=np.array([landmark2.x, landmark2.y, landmark2.z])
  base1=np.array([base1.x, base1.y, base1.z])
  base0=np.array([base0.x, base0.y, base0.z])

  segment_magnitude=math.dist(p1, p2)
  segment_direction = base1 - base0
  segment= segment_magnitude*normalize(segment_direction)

  r_alpha = R.from_rotvec(alpha * normalize(cross))

  rotated_vector_alpha = r_alpha.apply(segment)


  landmark_array=base1+rotated_vector_alpha 
  landmark=copy.copy(landmark2)
  
  landmark.x=landmark_array[0]
  landmark.y=landmark_array[1]
  landmark.z=landmark_array[2]

  #build landmark object?
  return landmark


def reconstruct_landmarks_from_angles(hand_landmarks, joint_angles, handedness):
  

  landmarks=[]
  for i in range(21):
    landmarks.append(hand_landmarks.landmark[mp_hands.HandLandmark(i)])
    
  reconstructed_hand_landmarks=copy.copy(landmarks)
  hand=1
  if handedness=="Right":
    hand=-1
  thumb_index_normal=-get_palm_normal(landmarks[1], landmarks[2], landmarks[5])
  index_middle_normal=-get_palm_normal(landmarks[0], landmarks[5], landmarks[9])
  middle_ring_normal=-get_palm_normal(landmarks[0], landmarks[9], landmarks[13])
  ring_pinky_normal=-get_palm_normal(landmarks[0], landmarks[13], landmarks[17])

  normals=[thumb_index_normal, index_middle_normal, middle_ring_normal, middle_ring_normal, ring_pinky_normal]

  for i in range(5):
    cross=get_cross(landmarks[0], landmarks[1+i*4], hand*normals[i])
    reconstructed_hand_landmarks[2+i*4]=reconstruct_pip(joint_angles[0+i*4], joint_angles[3+i*4], hand*normals[i], landmarks[0], landmarks[1+i*4], landmarks[2+i*4], handedness)
    reconstructed_hand_landmarks[3+i*4]=reconstruct_dip(joint_angles[1+i*4], hand*normals[i], landmarks[1+i*4], landmarks[2+i*4], landmarks[3+i*4], reconstructed_hand_landmarks[2+i*4], cross)
    reconstructed_hand_landmarks[4+i*4]=reconstruct_tip(joint_angles[2+i*4], hand*normals[i], reconstructed_hand_landmarks[2+i*4], landmarks[3+i*4], landmarks[4+i*4], reconstructed_hand_landmarks[3+i*4], cross)

  new_landmarks = landmark_pb2.NormalizedLandmarkList(landmark=reconstructed_hand_landmarks)

  return new_landmarks


#michael's synergies
manipulation_e1=normalize(np.array([[0.15075, 0.04867, -0.048864, 0.011474, 0.13871, 0.10185, 0.013991, 0.006332, 0.16573, 0.11729, 0.020231, 0.0096837, 0.15333, 0.176, 0.054559, -0.026265, 0.11766, 0.091763, 0.010513, -0.011513]])).T
manipulation_e2=normalize(np.array([[0.12939, -0.040741, -0.033309, -0.025255, 0.048632, -0.0051425, 0.0036897, -0.0029587, -0.025369, -0.0073484, -0.004823, -0.0055665, -0.058177, -0.02668, -0.0038935, 0.014417, -0.04606, -0.025093, 0.0017663, 0.0058283]])).T
manipulation_e3=normalize(np.array([[-0.126, -0.075822, -0.023047, 0.012231, -0.053607, 0.11037, 0.038378, 0.011935, -0.091651, 0.16981, 0.058706, 0.0050746, -0.099912, 0.2319, 0.10745, -0.022381, -0.071467, 0.12184, 0.018506, -0.036332]])).T
manipulation_e4=normalize(np.array([[0.07647, -0.048274, -0.024676, 0.039967, 0.15612, -0.02695, -0.00093293, -0.041477, 0.11587, -0.094454, -0.029349, -0.0036209, -0.079993, -0.01941, -0.013295, 0.048931, -0.18134, 0.061865, 0.009479, 0.050853]])).T

r2g_e1=normalize(np.array([0.16891, 0.025521, -0.032352, -0.0069859, 0.099574, 0.095142, 0.0073953, 0.0092749, 0.14129, 0.15698, 0.032075, 0.014296, 0.15207, 0.18345, 0.047436, -0.038703, 0.12477, 0.088426, 0.0053546, -0.027952])).T
r2g_e2=normalize(np.array([0.20385, -0.0092425, -0.041134, -0.0091354, 0.026853, 0.032833, 9.9717e-05, 0.0089803, -0.038916, -1.7963e-06, -0.0060575, 0.001091, -0.0018351, -0.034834, -0.035571, -0.011226, -0.02189, -0.055721, -0.019561, -0.014771])).T
r2g_e3=normalize(np.array([-0.026047, -0.05778, 0.132, -0.0090842, -0.067948, 0.023009, 0.01701, 0.043227, -0.17779, 0.13501, 0.056916, 0.0019381, -0.10527, 0.15653, 0.078045, -0.047216, -0.034363, 0.085201, 0.010366, -0.054433])).T
r2g_e4=normalize(np.array([-0.050509, -0.12138, 0.10046, -0.0080373, 0.23192, -0.0086001, 0.0097201, -0.071284, 0.19034, -0.086343, -0.031458, 0.0067945, -0.085191, -0.085097, -0.056322, 0.057298, -0.031216, 0.018081, 0.0027885, 0.086371])).T

synergies=np.hstack((manipulation_e1,manipulation_e2, manipulation_e3))

# For static images:
IMAGE_FILES = ['C:/Users/mxw00/Documents/meng/hand-interaction-capture/imgs/grasp.jpg']
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
      print("Handedness:", results.multi_handedness[idx].classification[0].label)
      joints=calc_joint_angles(hand_landmarks)

      lambdas = np.linalg.lstsq(synergies, joints, rcond=None)
  
      #use this to visualize the position to joints to position reconstruction
      new_landmarks=reconstruct_landmarks_from_angles(hand_landmarks, joints, results.multi_handedness[idx].classification[0].label)

      #use this to visualize the synergy reconstruction
      #new_landmarks=reconstruct_landmarks_from_angles(hand_landmarks, synergies@lambdas[0], results.multi_handedness[idx].classification[0].label)
      
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
      
      new_mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
      new_mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
          
      new_mp_drawing.draw_landmarks(
        annotated_image,
        new_landmarks,
        mp_hands.HAND_CONNECTIONS
      )
      #cv2.imshow("landmarks", annotated_image)
          
    cv2.imwrite(
        'C:/Users/mxw00/Documents/meng/hand-interaction-capture/imgs/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    # for hand_world_landmarks in results.multi_hand_world_landmarks:
    #   mp_drawing.plot_landmarks(
    #     hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
    #   new_mp_drawing.plot_landmarks(
    #     new_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
    

landmark_to_joint_angles=[]
# For webcam input:
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('cyberglove.mov')
lambdas_data=[]
set_synergies_flag=False

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
        landmark_to_joint_angles.append(calc_joint_angles(hand_landmarks))
        print("Confidence:", results.multi_handedness[idx].classification[0].score)

        mp_drawing.draw_landmarks(
          image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())

        if len(landmark_to_joint_angles)>=300:
          if not set_synergies_flag:
            set_synergies_flag=True
            np_landmarks=np.squeeze(np.array(landmark_to_joint_angles))
            print(np_landmarks.shape)

            u, s, vh = np.linalg.svd(np_landmarks, full_matrices=True)
            S = np.zeros((np_landmarks.shape[0], np_landmarks.shape[1]))
            S[:np_landmarks.shape[1], :np_landmarks.shape[1]] = np.diag(s)
            print(u.shape)
            print(S.shape)
            print(vh.shape)

          joints=calc_joint_angles(hand_landmarks)
        
          lambdas = np.linalg.lstsq(vh[:3].T, np.squeeze(joints), rcond=None)

          lambdas_data.append(lambdas[0])

          #new_landmarks=reconstruct_landmarks_from_angles(hand_landmarks, joints, results.multi_handedness[idx].classification[0].label)
          new_landmarks=reconstruct_landmarks_from_angles(hand_landmarks, vh[:3].T@(np.array([lambdas[0]]).T),  results.multi_handedness[idx].classification[0].label)

          new_mp_drawing.draw_landmarks(
            image,
            new_landmarks,
            mp_hands.HAND_CONNECTIONS
            )

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    #cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:

      break



cap.release()
print(np.array(lambdas_data).shape)

figure, axis = plt.subplots(1, 2)

axis[0].plot(lambdas_data)
axis[0].set_title("lambdas")
  
# For Cosine Function
axis[1].plot(np.diff(lambdas_data, axis=0))
axis[1].set_title("diff")

plt.show()

