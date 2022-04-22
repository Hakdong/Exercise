from this import d
import cv2
import mediapipe as mp
import time
import math
import numpy as np

#각도 계산
def calculate_angle(a,b,c) :
  
  a =np.array(a)
  b = np.array(b)
  c = np.array(c)
 
  radians = np.arctan2(c[2]-b[2],c[1]-b[1]) -np.arctan2(a[2]-b[2],a[1]-b[1]) 
  
  if exercise == 1:
    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) -np.arctan2(a[1]-b[1],a[0]-b[0]) 

  angle = np.abs(radians*180.0/np.pi)

  if angle>180.0 :
      angle = 360-angle
  
  return angle


# 운동 분류 
def  classification(shoulder,elbow,hand,hip,right_knee,foot,left_knee,left_shoulder) :

  global exercise
  shoulder =np.array(shoulder)
  elbow =np.array(elbow)
  hand =np.array(hand)
  hip =np.array(hip)
  right_knee =np.array(right_knee)
  foot =np.array(foot)
  left_knee =np.array(left_knee)
  left_shoulder=np.array(left_shoulder)
  arm_angle = calculate_angle(shoulder,elbow,hand)
  leg_angle = calculate_angle(hip,right_knee,foot)

  sh_vec = shoulder-left_shoulder
  knee_vec = right_knee-left_knee
  
  sh_dist = math.sqrt(sh_vec[0]**2)
  knee_dist = math.sqrt(knee_vec[0]**2)

  if hand[1] < shoulder[1] and  arm_angle> 150 : exercise =1 #pullup 
  if hand[1] > shoulder[1] and abs(hand[1]-shoulder[1]) >= abs(hip[1]-foot[1]) and hand[1] > right_knee[1]: exercise =2 #pushup
  if leg_angle < 120 and knee_dist > sh_dist*0.9  and hand[1] > shoulder[1] : exercise =3  #squat

  return exercise

# 오른손에 대한 회전 좌표
def rotation(center,a,theta):
  center = np.array(center)
  a = np.array(a)
  b = a -center

  new = [center[0]+math.cos(theta)*b[0]+math.sin(theta)*b[2],a[1],center[2]-math.sin(theta)*b[0]+math.cos(theta)*b[2]]
  return new

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
number = [0,0,0] #[0]은 운동수행확인변수 [1]은 횟수 
# object_reps = int(input("원하는 횟수를 적어주세요::"))
# object_sets = int(input("원하는 세트수를 적어주세요::"))
object_number =[3,3] #[0]은 수행횟수 [1] 수행세트횟수3
time_number =0 # 타이머 변수
kindofexercise = "none"
exercise = 0
# time_input = int(input("휴식시간을 설정하세요(단위s)::"))

cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
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
    results = pose.process(image)
    #change video size
    image = cv2.resize(image,(1024,1024))
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    landmarks = results.pose_landmarks.landmark 
    
    left_hand =[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
    right_hand =[landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x ,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z ]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x ,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z ]      
    right_hip =[landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x ,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y ,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z ]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x  ,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y ,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
    right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x ,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y  ,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z]  
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x ,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y ,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x  ,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y ,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
    minute = time_number /60
    second = time_number %60 
    theta = np.arctan((left_hand[2]-right_hand[2])/(left_hand[0]-right_hand[0]))


    new_left_shoulder = rotation(right_hand,left_shoulder,theta)
    new_left_hand = rotation(right_hand,left_hand,theta)
    new_right_shoulder = rotation(right_hand,right_shoulder,theta)
    new_right_hip = rotation(right_hand,right_hip,theta)
    new_right_knee = rotation(right_hand,right_knee,theta)
    new_right_foot = rotation(right_hand,right_foot,theta)
    new_right_elbow = rotation(right_hand,right_elbow,theta)
    new_left_knee = rotation(right_hand,left_knee,theta)

    exercise = classification(new_right_shoulder,new_right_elbow,right_hand,new_right_hip,new_right_knee,new_right_foot,new_left_knee,new_left_shoulder) 
    
    if exercise == 0 :
      kindofexercise ="none" 
      object_number =[3,3]
    if exercise == 1 :   #pull_up
      kindofexercise ="pullup"   
      angle = calculate_angle(right_shoulder,right_elbow,right_hand)
      if angle < 45 and time_number ==0:
        number[0] = 1
      if angle > 150 and number[0] == 1 and time_number ==0:
        number[1] += 1
        number[0] = 0
      cv2.putText(
      image, str(angle), tuple(np.multiply([right_elbow[0],right_elbow[1]],[1024,1024]).astype(int)),
      cv2.FONT_HERSHEY_SIMPLEX, 1,
      color=(0,255,0), thickness=3)
    if exercise == 2 : #push_up
      kindofexercise = "pushup"
      angle = calculate_angle(new_right_shoulder,new_right_elbow,right_hand)
      if angle < 90 and time_number ==0:
        number[0] = 1
      if angle > 140 and number[0] == 1 and time_number ==0:
        number[1] += 1
        number[0] = 0
      cv2.putText(
      image, str(angle), tuple(np.multiply([right_elbow[0],right_elbow[1]],[1024,1024]).astype(int)),
      cv2.FONT_HERSHEY_SIMPLEX, 1,
      color=(0,255,0), thickness=3)
    if exercise == 3 : #squat
      kindofexercise = "squat"
      angle = calculate_angle(new_right_hip,new_right_knee,new_right_foot)
      if angle<100 and time_number ==0:
        number[0] = 1
      if angle > 150 and number[0] == 1 and time_number ==0:
        number[1] += 1
        number[0] = 0
      cv2.putText(
      image, str(angle), tuple(np.multiply([right_knee[0],right_knee[1]],[1024,1024]).astype(int)),
      cv2.FONT_HERSHEY_SIMPLEX, 1,
      color=(0,255,0), thickness=3)

    cv2.putText(
      image, text='%s reps=%dset=%d rest %dm %ds' % (kindofexercise,object_number[0]-number[1],
      object_number[1]-number[2],minute,second), org=(10, 30),
      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
      color=(255,0,255), thickness=3)


    # cv2.putText(
    #   image, text='%f %f %f %f %f %f' % (new_right_shoulder[1],new_right_shoulder[2],new_right_elbow[1],new_right_elbow[2],
    #    right_hand[1],right_hand[2]), org=(10, 30),
    #   fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
    #   color=(255,0,0), thickness=3) 
    
    if number[1] == object_number[0] and time_number ==0 :
      number[2] += 1 
      number[1] = 0 
      time_number = 10
    if number[2] == object_number[1] and time_number ==0:
       number[2] = 0
       kindofexercise = "none"
       exercise = 0
       print("목표한 운동세트를 모두 끝냈습니다.고생하셨습니다.")

    if time_number !=0 :
      start = time.time() 
      if start % 1 > 0.90 :
        time_number -=1
      if cv2.waitKey(5) == ord('p'):
        time_number =0

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == ord('s') :
      exercise =0
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()