import cv2
import threading
import mediapipe as mp
import numpy as np
import vg
import math
import copy
import matplotlib.pyplot as plt
from mediapipe.framework.formats import landmark_pb2
from scipy.spatial.transform import Rotation as R
import yaml
from types import SimpleNamespace

import time

from scipy.spatial.transform import Rotation as R
import pandas as pd
import os

import torch
import torch.multiprocessing
import recording.util as util
from scipy.spatial.distance import cdist



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

new_mp_drawing = mp.solutions.drawing_utils

file_path = "C:/Users/Maggie/hand-interaction-capture"

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


##############################################################

def get_image_coordinate(x, y, shape): 
    
    relative_x = int(np.clip(x, 0, 1) * (shape[0]-1))
    relative_y = int(np.clip(y, 0, 1) * (shape[1]-1))
    return relative_x, relative_y

def cam_preview(previewName, camID):
    best_model = torch.load(util.find_latest_checkpoint(config))
    best_model.eval()

    landmark_to_joint_angles_right=[]
    landmark_to_joint_angles_left=[]

    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cam.set(cv2.CAP_PROP_FPS, 30)

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
                scale = 355
                image = crop_and_resize(image, scale)
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        if get_palm_area(hand_landmarks)>=0.01:
                            handedness = results.multi_handedness[idx].classification[0].label
                            if handedness=='Left':
                                landmark_to_joint_angles_right.append(np.insert(calc_joint_angles(hand_landmarks).T.flatten(), 0, [time.time(), results.multi_handedness[idx].classification[0].score, get_palm_area(hand_landmarks)]))
                            elif handedness=='Right':
                                landmark_to_joint_angles_left.append(np.insert(calc_joint_angles(hand_landmarks).T.flatten(), 0, [time.time(), results.multi_handedness[idx].classification[0].score, get_palm_area(hand_landmarks)]))
                        
                        joints=calc_joint_angles(hand_landmarks)

                        
                        force_pred = run_model(image, best_model)

                        shape = image.shape 
                        fingertips=[4, 8, 12, 16, 20]
                        contact=[0, 0, 0, 0, 0]
                        if np.any(force_pred):
                            
                            for i, landmark in enumerate(hand_landmarks.landmark):
                                if i in fingertips:
                                    x, y = get_image_coordinate(landmark.y, landmark.x, shape)
                                    if np.min(cdist([[x, y]],np.transpose(np.nonzero(force_pred))))<20:
                                        #print(i, ": ", force_pred[x][y])
                                        # how to get pressure estimate??
                                    

                    
                        #use this to visualize the position to joints to position reconstruction
                        new_landmarks=reconstruct_landmarks_from_angles(hand_landmarks, joints, results.multi_handedness[idx].classification[0].label)

                        image = cv2.addWeighted(image, 0.7, util.pressure_to_colormap(force_pred), 1.0, 0.0)
                        #use this to visualize the synergy reconstruction
                        #lambdas = np.linalg.lstsq(synergies, joints, rcond=None)
                        #new_landmarks=reconstruct_landmarks_from_angles(hand_landmarks, synergies@lambdas[0], results.multi_handedness[idx].classification[0].label)
      
                        ### if you want to visualize the joint angles, you can uncomment this
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks, #hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                        #use this to overlay synergy reconstruction
                        # new_mp_drawing.draw_landmarks(
                        #     image,
                        #     new_landmarks,
                        #     mp_hands.HAND_CONNECTIONS
                        #     )

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



disp_x = 480 * 2
disp_y = 384 * 2
aspect_ratio = 480 / 384


def run_model(img, best_model):
    # Takes in a cropped OpenCV-formatted image, does the preprocessing, runs the network, and the postprocessing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255
    img = util.resnet_preprocessor(img)
    img = img.transpose(2, 0, 1).astype('float32')
    img = torch.tensor(img).unsqueeze(0)

    with torch.no_grad():
        force_pred_class = best_model(img.cuda())
        force_pred_class = torch.argmax(force_pred_class, dim=1)
        force_pred_scalar = util.classes_to_scalar(force_pred_class, config.FORCE_THRESHOLDS)
        force_pred_scalar = force_pred_scalar.detach().squeeze().cpu().numpy()

    return force_pred_scalar


def crop_and_resize(img, scale):
    y_scale = int(scale / aspect_ratio)

    start_x_int = max(img.shape[1] // 2 - scale, 0)
    end_x_int = min(img.shape[1] // 2 + scale, img.shape[1])
    start_y_int = max(img.shape[0] // 2 - y_scale, 0)
    end_y_int = min(img.shape[0] // 2 + y_scale, img.shape[0])
    crop_frame = img[start_y_int:end_y_int, start_x_int:end_x_int, :]

    resize_frame = cv2.resize(crop_frame, (480, 384))

    return resize_frame

if __name__ == "__main__":

    with open('./config/paper.yml', 'r') as stream:
        data = yaml.safe_load(stream)

    data_obj = SimpleNamespace(**data)
    data_obj.CONFIG_NAME = 'paper'
    config = data_obj

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_cam_threads(2)



   






