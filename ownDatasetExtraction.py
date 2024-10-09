import cv2
import dlib
import numpy as np
from imutils import face_utils
import csv
import os
import imutils

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

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
    


with open('ownDatasetCheck.csv', 'w', newline='') as csvfile:

    fieldnames = ['Frame', 'EAR', 'DROWSY', 'SIDE_TILT', 'FRONT_TILT']
    csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
    csvwriter.writeheader()

    video_directory = '/Users/thanan/Thananjayan/VITSEM6/AI/AI Project Final/Videos/'

    for filename in os.listdir(video_directory):
        print(filename)
        if 'drowsy' in filename:
            drowsyClass = 1
        else:
            drowsyClass = 0
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            video_file = os.path.join(video_directory, filename)
    
            cap = cv2.VideoCapture(video_file)
    
            if not cap.isOpened():
                print(f"Error: Unable to open video file {filename}")
                continue
    
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
            frames_to_analyze = 2
    
            while cap.isOpened():
                ret, frame = cap.read()
    
                if not ret:
                    break
    
                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
                if frame_number % int(frame_rate / frames_to_analyze) == 0:
                    frame = imutils.resize(frame, width=450)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    subjects = detect(gray, 0)
                    for subject in subjects:
                        shape = predict(gray, subject)
                        shape = face_utils.shape_to_np(shape)
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
                        
                        csvwriter.writerow({'Frame': frame_number, 'EAR': ear, 'DROWSY': drowsyClass, 'SIDE_TILT': sideAngle, 'FRONT_TILT': frontAngle})
    
            cap.release()