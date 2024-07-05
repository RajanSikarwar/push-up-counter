import cv2
import mediapipe as md

md_drawing = md.solutions.drawing_utils
md_drawing_styles = md.solutions.drawing_styles
md_pose = md.solutions.pose

count = 0

position=None
#taking input from camera
cap = cv2.VideoCapture(0)

#importing pose
with md_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as pose:
    #checking for camera input
    while cap.isOpened():
        success,image = cap.read()
        if not success:
            print("empty camera")
            break
    #processing the image by flipping it completely using value 1
    #original input is BGR but we want RGB for mediapipe
        image = cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB)
        result = pose.process(image)

    #creating a list for all 34 points of the pose package of mediapipe
        imlist = []

    #if condition checking if any "body" is present in the camera in put
        if result.pose_landmarks:
            md_drawing.draw_landmarks(
                image, #image on which we want to draw
                result.pose_landmarks, #these are the points 
                md_pose.POSE_CONNECTIONS #these are the lines
            )
        #enum returns and id number for the landmark and a list of the ratio of coordinates
        #we will multiply the ratio with the image absolute height and width to get the exact location
        #and append the id number and x and y coordinates into our list 
            for id,im in enumerate(result.pose_landmarks.landmark):
                h,w,_ = image.shape
                X,Y = int(im.x*w),int(im.y*h)
                imlist.append([im,X,Y])

    #shoulder points are 11 and 12 and elbows are 13 and 14 
    #logic used to count the push up is if the shoulders reach below the elbow
    #after that again reach above the elbow its count is incremented as +1
        if len(imlist) != 0: #checking length if there is no one in front of camera its 0
            if ((imlist[12][2] - imlist[14][2])>=15 and (imlist[11][2] - imlist[13][2])>=15):
                position = "down"
            if ((imlist[12][2] - imlist[14][2])<=5 and (imlist[11][2] - imlist[13][2])<=5) and position == "down":
                position = "up"
                count +=1 
                print(count) 

        cv2.imshow("Push-up counter",cv2.flip(image,1))
        key=cv2.waitKey(1)
        if key==ord('q'):
            break

cap.release()