import cv2
import numpy as np
import math
from typing import NamedTuple
from collections import deque, namedtuple

import mediapipe as mp

capture_hands = mp.solutions.hands.Hands()
drawing_option = mp.solutions.drawing_utils

hand_keypoints = {
    0: "WRIST",
    1: "THUMB_CMC",
    2: "THUMB_MCP",
    3: "THUMB_IP",
    4: "THUMB_TIP",
    5: "INDEX_FINGER_MCP",
    6: "INDEX_FINGER_PIP",
    7: "INDEX_FINGER_DIP",
    8: "INDEX_FINGER_TIP",
    9: "MIDDLE_FINGER_MCP",
    10: "MIDDLE_FINGER_PIP",
    11: "MIDDLE_FINGER_DIP",
    12: "MIDDLE_FINGER_TIP",
    13: "RING_FINGER_MCP",
    14: "RING_FINGER_PIP",
    15: "RING_FINGER_DIP",
    16: "RING_FINGER_TIP",
    17: "PINKY_MCP",
    18: "PINKY_PIP",
    19: "PINKY_DIP",
    20: "PINKY_TIP",
}
CURRENT_FRAME_WEIGHT = 0.75
def smooth_landmarks(list_of_hands: list[NamedTuple]):
    """ Returns a Hand Object with landmarks calculated with a weighted moving average"""
    list_of_hands = [hand for hand in list_of_hands if hand is not None]
    if len(list_of_hands) == 1:
        return list_of_hands[0]
    list_of_hand_landmarks = [hand.landmark for hand in list_of_hands]
    list_of_landmarks = []
    for landmark in zip(*list_of_hand_landmarks):
        number_of_datapoints = len(landmark)
        previous_frame_weight = (1-CURRENT_FRAME_WEIGHT)/(number_of_datapoints - 1)
        average_x = previous_frame_weight * sum(point.x for point in landmark[:-1])
        average_y = previous_frame_weight * sum(point.y for point in landmark[:-1])
        average_x += CURRENT_FRAME_WEIGHT * landmark[-1].x
        average_y += CURRENT_FRAME_WEIGHT * landmark[-1].y
        Point = namedtuple('Point', ['x', 'y'])
        list_of_landmarks.append(Point(average_x, average_y))

    Hand = namedtuple('Hand', ['landmark'])
    hand = Hand(list_of_landmarks)
    return hand
    
def hand_distance(hand_1: NamedTuple, hand_2: NamedTuple) -> float:
    """Given two hand objects, returns the sum of the distance between all the
    landmarks"""
    total_distance = 0
    hand_1_landmarks = hand_1.landmark
    hand_2_landmarks = hand_2.landmark
    for h1_landmark, h2_landmark in zip(hand_1_landmarks, hand_2_landmarks):
        x_1, y_1 = h1_landmark.x, h1_landmark.y
        x_2, y_2 = h2_landmark.x, h2_landmark.y
        distance = math.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)
        total_distance += distance
    return total_distance


def circle_tips(frame: np.ndarray, hand: NamedTuple):
    """Circles the tips of fingers in a given frame"""
    landmarks = hand.landmark
    frame_height, frame_width, _ = frame.shape
    for id, landmark in enumerate(landmarks):
        x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
        tips = [
            "THUMB_TIP",
            "INDEX_FINGER_TIP",
            "MIDDLE_FINGER_TIP",
            "RING_FINGER_TIP",
            "PINKY_TIP",
        ]
        if hand_keypoints[id] in tips:
            cv2.circle(frame, (x, y), 10, (0, 255, 0))

    return frame


def get_hand(previous_hand: NamedTuple, all_hands: list):
    """Returns either the first hand detected or the hand closet"""
    if previous_hand is None or len(all_hands) == 0:
        hand = all_hands[0]
    else:
        hands_sorted_by_distance = sorted(
            all_hands, key=lambda x: hand_distance(x, previous_hand)
        )
        hand = hands_sorted_by_distance[0]

    return hand

def process_frame(frame: np.ndarray, previous_hands: NamedTuple):
    output_hands = capture_hands.process(frame)
    all_hands = output_hands.multi_hand_landmarks
    is_hand_inframe = False
    previous_hand = previous_hands[-1]
    if all_hands:
        hand = get_hand(previous_hand, all_hands)
        is_hand_inframe = True

    if not is_hand_inframe:
        cv2.imshow("", frame)
        return [None]
    hand = smooth_landmarks(previous_hands + [hand])
    #drawing_option.draw_landmarks(frame, hand)
    frame = circle_tips(frame, hand)

    cv2.imshow("", frame)
    previous_hands.append(hand)
    if len(previous_hands) > 10:
        previous_hands.pop(0)
    return previous_hands


def main():
    cap = cv2.VideoCapture(0)
    previous_hand = [None]
    while True:
        ret, frame = cap.read()
        if ret:
            previous_hand = process_frame(frame, previous_hand)
        else:
            print("No frame")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


main()
