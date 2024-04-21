import cv2
import numpy as np
import math

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


capture_hands = mp.solutions.hands.Hands()
drawing_option = mp.solutions.drawing_utils

def hand_distance(hand_1, hand_2):
    total_distance = 0
    hand_1_landmarks = hand_1.landmark
    hand_2_landmarks = hand_2.landmark
    for h1_landmark, h2_landmark in zip(hand_1_landmarks, hand_2_landmarks):
        x_1, y_1 = h1_landmark.x, h1_landmark.y
        x_2, y_2 = h2_landmark.x, h2_landmark.y
        distance = math.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)
        total_distance += distance
    print(total_distance)
    return total_distance

def process_frame(frame: np.ndarray, previous_hand):
    output_hands = capture_hands.process(frame)
    all_hands = output_hands.multi_hand_landmarks
    is_hand_inframe = False
    if all_hands:
        if previous_hand is None or len(all_hands) == 0:
            hand = all_hands[0]
        else:
            hands_sorted_by_distance = sorted(all_hands, key=lambda x: hand_distance(x, previous_hand))
            hand = hands_sorted_by_distance[0]

        is_hand_inframe = True

    if not is_hand_inframe:
        cv2.imshow("", frame)
        print("no hand")
        return None
    
    drawing_option.draw_landmarks(frame, hand)
    landmarks = hand.landmark
    for id, landmark in enumerate(landmarks):
        x, y = int(landmark.x), int(landmark.y)
        if id == 8:
            cv2.circle(frame,(x,y),10,(0,255,255))
        


    cv2.imshow("", frame)
    #print('hand',hand)
    return hand


def main():
    cap = cv2.VideoCapture(0)
    mpDraw = mp.solutions.drawing_utils
    previous_hand = None
    while True:
        ret, frame = cap.read()
        if ret:
            previous_hand = process_frame(frame, previous_hand)
            #print(previous_hand is not None)
        else:
            print("No frame")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # time.sleep(0.001)

    cap.release()
    cv2.destroyAllWindows()


main()
