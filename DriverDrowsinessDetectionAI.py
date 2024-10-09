from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import numpy as np
import time
import os.path
import pickle
import warnings

warnings.filterwarnings('ignore')

mixer.init()
mixer.music.load("alert.wav")
blink_start_time = time.time()
intervalStart = time.time()
blink_count = 0


if os.path.exists("trained_Random Forest.pickle"):
    print("Loading Trained Model")
    model = pickle.load(open("trained_Random Forest.pickle", "rb"))
    print(model)
    
else:
    print("Creating and training a new model")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    lipDistance = abs(top_mean[1] - low_mean[1])
    return lipDistance

def side_tilt(leftEye, rightEye):
    left_eye_x = (leftEye[1][0] + leftEye[5][0])/2
    right_eye_x = (rightEye[1][0] + rightEye[5][0])/2
    left_eye_y = (leftEye[3][1]+leftEye[0][1])/2
    right_eye_y = (rightEye[3][1] + rightEye[0][1])/2
    
    delta_x = right_eye_x - left_eye_x 
    delta_y = right_eye_y - left_eye_y 
    
    angle = np.arctan(delta_y / delta_x)   
    angle = (angle * 180) / np.pi  
    return angle

def front_tilt(image_points, size):
    model_points = np.array([
        (0.0, 0.0, 0.0),             
        (0.0, -330.0, -65.0),        
        (-225.0, 170.0, -135.0),     
        (225.0, 170.0, -135.0),      
        (-150.0, -150.0, -125.0),    
        (150.0, -150.0, -125.0)      
        ])

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
        ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)     
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    
    axis_points, _ = cv2.projectPoints(np.float32([(0, 0, 0),
                                                    (0, 0, -100),
                                                    (0, 100, 0),
                                                    (100, 0, 0)]), rotation_vector, translation_vector,
                                       camera_matrix, dist_coeffs)

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    theta_x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

    theta_x_deg = np.degrees(theta_x)
    if theta_x_deg > 0:
        return 180 - theta_x_deg
    else:
        return 180 + theta_x_deg

frame_check = 20
yawnThresh = 20

detect = dlib.get_frontal_face_detector()

predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap=cv2.VideoCapture(0)
flag=0
yawnFlag=0
tiltFlag=0
while True:
    ret, frame=cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    elapsed_time = time.time() - intervalStart

    for subject in subjects:
        shape1 = predict(gray, subject)
        shape = face_utils.shape_to_np(shape1)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        
        sideAngle = side_tilt(leftEye, rightEye)
        
        size = frame.shape
        image_points = np.array([
            (shape[33][0], shape[33][1]),     
            (shape[8][0], shape[8][1]),       
            (shape[36][0], shape[36][1]),     
            (shape[45][0], shape[45][1]),     
            (shape[48][0], shape[48][1]),     
            (shape[54][0], shape[54][1])      
            ], dtype="double")
        frontAngle = front_tilt(image_points, size)
        
        frameData = [[ear, sideAngle, frontAngle]]
        print(frameData)
        
        drowsyState = model.predict(frameData)
       
        lipDistance = lip_distance(shape)
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)
        for n in range(0, 68):
        	(x,y) = shape[n]
        	cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
            
        if drowsyState == 1:
            flag += 1
            if flag >= frame_check:
                cv2.putText(frame, "You are drowsy. Take rest", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
        else:
            flag = 0
            
        if (lipDistance > yawnThresh):
            yawnFlag += 1
            if yawnFlag == 3:
                frame_check = max(16, min(round(frame_check-2), 27))
                print("Frames: ")
                print(frame_check)
           
        if sideAngle > 15 or sideAngle < -15:    
            if sideAngle > 0:
                print("left")
                #cv2.putText(frame, 'LEFT TILT :' + str(int(sideAngle))+' degrees', 
                           #(10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                print("right")
                #cv2.putText(frame, 'RIGHT TILT :' + str(int(sideAngle))+' degrees', 
                           #(10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else: 
            print("straight")
            #cv2.putText(frame, 'STRAIGHT', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Checking blink rate
        if ear < 0.18:  
            if time.time() - blink_start_time > 1.5:  
                blink_count += 1
                blink_start_time = time.time()

        if elapsed_time > 60:
            print(f'interval over. blinks: {blink_count}')
            if blink_count <= 6:
                print("As blinks are <6, adjusting threshold and frame_check value")
                if yawnFlag>2:
                    frame_check = max(16, min(round(frame_check-2), 27))
            elif yawnFlag<3:
                frame_check = max(16, min(round(frame_check+2), 27))
            blink_count = 0
            intervalStart = time.time()

            
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()