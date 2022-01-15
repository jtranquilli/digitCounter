import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands #convention 
hands = mpHands.Hands() #empty since we're using default parameters
mpDraw = mp.solutions.drawing_utils
handsList = []
result = 0

def setdigits(hand):
    hand.digits = hand.a + hand.b + hand.c + hand.d + hand.e
    
def getdigits(hand):
    return hand.digits

def  setresult(result):
    d = []
    val = 0
    for x in handsList:
        d.append(getdigits(x))
    for i in d:
        val+= i
    
class hand:
    def __init__(self,myid, a, b, c, d, e):
        
        self.myid = myid
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.digits = a + b + c + d + e
     


while True:
    success, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(imgRGB)
        
    if results.multi_hand_landmarks: #if anything is detected
        
        
        for id, handLms in enumerate(results.multi_hand_landmarks): #handLms is a single hand
            #hand objects are stored in results.multi_hand_landmarks like an array
            
            
            newhand = hand(id,0,0,0,0,0)
            handsList.append(newhand) #id corresponds to index in handsList
            
            
            
            for id, lm in enumerate(handLms.landmark): #enumerating landmarks
                
                h, w, c = img.shape
                
                #cx, cy = int(lm.x*w), int(lm.y*h)
                #height width and channels of image
                
                thumbTip = handLms.landmark[mpHands.HandLandmark.THUMB_TIP].y * h * (-1) + 700
                
                indexTip = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * h * (-1) + 700
                
                midFingerTip = handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].y * h * (-1) + 700
                
                ringFingerTip = handLms.landmark[mpHands.HandLandmark.RING_FINGER_TIP].y * h * (-1) + 700
                
                pinkyTip = handLms.landmark[mpHands.HandLandmark.PINKY_TIP].y * h * (-1) + 700
                
                wrist = handLms.landmark[mpHands.HandLandmark.WRIST].y * h * (-1) + 700
                
                indexDip = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_DIP].y * h * (-1) + 700
                
                midDip = handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_DIP].y * h * (-1) + 700
                
                ringDip = handLms.landmark[mpHands.HandLandmark.RING_FINGER_DIP].y * h * (-1) + 700
                
                pinkyDip = handLms.landmark[mpHands.HandLandmark.PINKY_DIP].y * h * (-1) + 700
                
                thumbIp = handLms.landmark[mpHands.HandLandmark.THUMB_IP].y * h * (-1) + 700
                
                indexMcp = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP].y * h * (-1) + 700
                
                indexPip = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_PIP].y * h * (-1) + 700
                
                midPip = handLms.landmark[mpHands.HandLandmark.MIDDLE_FINGER_PIP].y * h * (-1) + 700
                
                if (indexTip > indexDip and indexDip > indexPip):
                    newhand.a = 1
                    setdigits(newhand)
                    
                if(midFingerTip > midDip and midDip > midPip):
                    newhand.b = 1
                    setdigits(newhand)
                    
                if (ringFingerTip > ringDip):
                    newhand.c = 1
                    setdigits(newhand)   
                    
                if (pinkyTip > pinkyDip):
                    newhand.d = 1
                    setdigits(newhand)
                    
                    
                #dealing with the thumb requires looking at the x axis to know whether it is tucked into the hand  
                #compare absolute value of difference in x coordinates for thumb tip to pinkyMCP with indexMCP to pinkyMCP
                
                thumbTipx = handLms.landmark[mpHands.HandLandmark.THUMB_TIP].x * h * (-1) + 700
                indexFingerMCPx = handLms.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP].x * h * (-1) + 700
                pinkyFingerMCPx = handLms.landmark[mpHands.HandLandmark.PINKY_MCP].x * h * (-1) + 700
                
                #absolute distance between indexMCPx and pinkyMCPx
                #if the thumbtip is closer we can consider the thumb as being tucked in
                
                
                thumbPinkyDist = abs(pinkyFingerMCPx - thumbTipx)
                
                flagDist = abs(pinkyFingerMCPx -  indexFingerMCPx)
                
                
                if(thumbPinkyDist > flagDist):
                    newhand.e = 1
                    setdigits(newhand)
                    #thumbTip > thumbIp and thumbTip > indexMcp and
                    
                    
                print(newhand.digits)
                
                cv2.putText(img, str(newhand.digits),(10,70), cv2.FONT_HERSHEY_PLAIN,4,(255,0,255), 3)
             
                
            mpDraw.draw_landmarks(img, handLms)#, mpHands.HAND_CONNECTIONS)
            #simply remove the last parameter to stop illustrating lines
    
    
    cv2.imshow("Image", img) #display
    
    cv2.waitKey(1)

