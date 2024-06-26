import cv2
import numpy as np
import math
from typing import NamedTuple
import pyautogui
import mediapipe as mp

#pyautogui.FAILSAFE = False

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
capture_hands = mp.solutions.hands.Hands()
drawing_option = mp.solutions.drawing_utils

NUMBER_OF_PREVIOUS_HANDS = 10
ALPHA = 0.4
SENSITIVITY = 2.5


class Point(NamedTuple):
    x: int | float
    y: int | float


class HandCopy(NamedTuple):
    landmark: list

x_data = []
x_data_smooth = []

def smooth_landmarks(list_of_hands: list[NamedTuple]) -> HandCopy:
    """Returns a Hand Object with landmarks calculated with a weighted moving average"""
    global x_data
    global x_data_smooth
    list_of_hands = [hand for hand in list_of_hands if hand is not None]
    if len(list_of_hands) == 1:
        return list_of_hands[0]
    list_of_hand_landmarks = [hand.landmark for hand in list_of_hands]
    list_of_landmarks = []
    landmark_count = 0
    for landmark in zip(*list_of_hand_landmarks):
        datapoints = list(reversed(landmark))
        average_x = 0
        average_y = 0
        weights = [(1 - ALPHA) ** i for i in range(len(datapoints))]
        for i in range(len(datapoints)):
            average_x += weights[i] * datapoints[i].x
            average_y += weights[i] * datapoints[i].y

        average_x = average_x / sum(weights)
        average_y = average_y / sum(weights)
        if landmark_count == 0:
            x_data.append(datapoints[0].x)
            x_data_smooth.append(average_x)
        list_of_landmarks.append(Point(average_x, average_y))
        landmark_count += 1

    return HandCopy(list_of_landmarks)


def distance(x_1, y_1, x_2, y_2):
    return math.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)


def hand_distance(hand_1: NamedTuple, hand_2: NamedTuple) -> float:
    """Given two hand objects, returns the sum of the distance between all the
    landmarks"""
    total_distance = 0
    hand_1_landmarks = hand_1.landmark
    hand_2_landmarks = hand_2.landmark
    for h1_landmark, h2_landmark in zip(hand_1_landmarks, hand_2_landmarks):
        x_1, y_1 = h1_landmark.x, h1_landmark.y
        x_2, y_2 = h2_landmark.x, h2_landmark.y
        total_distance += distance(x_1, y_1, x_2, y_2)
    return total_distance


def circle_landmarks(frame: np.ndarray, hand: NamedTuple) -> np.ndarray:
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
            "INDEX_FINGER_MCP",
        ]
        if hand_keypoints[id] in tips:
            cv2.circle(frame, (x, y), 10, (0, 255, 0))

    return frame


def get_hand(previous_hand: NamedTuple, all_hands: list) -> NamedTuple:
    """Returns either the first hand detected or the hand closet"""
    if previous_hand is None or len(all_hands) == 0:
        hand = all_hands[0]
    else:
        hands_sorted_by_distance = sorted(
            all_hands, key=lambda x: hand_distance(x, previous_hand)
        )
        hand = hands_sorted_by_distance[0]

    return hand


def distance(x_1, y_1, x_2, y_2):
    return math.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)


def move_cursor(hand: HandCopy, previous_hand: HandCopy):
    keypoint_to_index = {v: k for k, v in hand_keypoints.items()}
    cursor_point = keypoint_to_index["INDEX_FINGER_MCP"]
    delta_x = hand.landmark[cursor_point].x - previous_hand.landmark[cursor_point].x
    delta_y = hand.landmark[cursor_point].y - previous_hand.landmark[cursor_point].y
    screen_width, screen_height = pyautogui.size()
    x_scaler = screen_width
    y_scaler = screen_height
    delta_x *= -x_scaler * SENSITIVITY
    delta_y *= y_scaler * SENSITIVITY
    # print(delta_x, delta_y)
    if (
        abs(delta_x) > SENSITIVITY and abs(delta_y) > SENSITIVITY
    ):  # Threshold to prevent jittering
        pyautogui.move(delta_x, delta_y, _pause=False)

frame_counter = 0
move_mouse = 0
left_click_count = 0
right_click_count = 0
def process_landmark_data(
    hand: HandCopy, previous_hand: HandCopy, actions: list = []
) -> None:
    global frame_counter
    global move_mouse
    global left_click_count
    global right_click_count
    """Takes the hand and runs actions based on the distance between landmarks."""
    keypoint_to_index = {v: k for k, v in hand_keypoints.items()}
    index_finger_index = keypoint_to_index["INDEX_FINGER_TIP"]
    thumb_index = keypoint_to_index["THUMB_TIP"]
    middle_finger_index = keypoint_to_index["MIDDLE_FINGER_TIP"]
    ring_finger_index = keypoint_to_index["RING_FINGER_TIP"]
    index_tip_coordinates = (
        hand.landmark[index_finger_index].x,
        hand.landmark[index_finger_index].y,
    )
    thumb_tip_coordinates = (hand.landmark[thumb_index].x, hand.landmark[thumb_index].y)
    middle_finger_coordinates = (
        hand.landmark[middle_finger_index].x,
        hand.landmark[middle_finger_index].y,
    )
    ring_finger_coordinates = (
        hand.landmark[ring_finger_index].x,
        hand.landmark[ring_finger_index].y
    )
    #print(distance(*index_tip_coordinates, *thumb_tip_coordinates))
    print(distance(*middle_finger_coordinates, *thumb_tip_coordinates))

    if distance(*index_tip_coordinates, *thumb_tip_coordinates) < 0.1:
        move_cursor(hand, previous_hand)
        print("MOVING CURSOR")
        move_mouse += 1

    if (
        distance(*middle_finger_coordinates, *thumb_tip_coordinates) < 0.08
        #and "left_click" not in actions
    ):
        #pyautogui.mouseDown(_pause=False)
        actions.append("left_click")
        print("LEFT CLICK")
        left_click_count += 1
    else:
        if (
            distance(*middle_finger_coordinates, *thumb_tip_coordinates) > 0.1
            and "left_click" in actions
        ):
            pyautogui.mouseUp(_pause=False)
            actions.remove("left_click")

    if (
        distance(*ring_finger_coordinates, *thumb_tip_coordinates) < 0.1
        #and "right_click" not in actions
    ):
        #pyautogui.rightClick(_pause=False)
        actions.append("right_click")
        print("RIGHT CLICK")
        right_click_count += 1
    else:
        if (
            distance(*ring_finger_coordinates, *thumb_tip_coordinates) > 0.1
            and "right_click" in actions
        ):
            actions.remove("right_click")

    frame_counter += 1

    return actions



def process_frame(
    frame: np.ndarray, previous_hands: list, actions: list = [],get_results=False
) -> NamedTuple:
    """Process the frames and collects landmark data. Returns a list of previous hands
    to be passed onto the next call."""
    output_hands = capture_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    all_hands = output_hands.multi_hand_landmarks
    is_hand_inframe = False
    previous_hand = previous_hands[-1]
    if all_hands:
        hand = get_hand(previous_hand, all_hands)
        is_hand_inframe = True

    if not is_hand_inframe:
        cv2.imshow("", frame)
        return [None], []
    drawing_option.draw_landmarks(frame, hand)
    hand = smooth_landmarks(previous_hands + [hand])
    frame = circle_landmarks(frame, hand)
    if previous_hand is not None:
        actions = process_landmark_data(hand, previous_hand)
    cv2.imshow("", frame)
    previous_hands.append(hand)
    if len(previous_hands) > NUMBER_OF_PREVIOUS_HANDS:
        previous_hands.pop(0)
    return previous_hands, actions


def main():
    cap = cv2.VideoCapture("right_click.mp4")
    previous_hand = [None]
    actions = []
    while True:
        ret, frame = cap.read()
        if ret:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            frame = cv2.filter2D(frame, -1, kernel)
            previous_hand, actions = process_frame(frame, previous_hand, actions)
        else:
            print("No frame")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(x_data_smooth)
    print(x_data)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_xlabel("Frame Number (t)")
    ax.set_ylabel("X position of landmark")
    ax.plot(x_data[100:200], label="X")
    ax.plot(x_data_smooth[100:200], label="X with Exponential Smoothing")
    ax.legend()

    ax.set_title("Exponential Smoothing on X position of landmarks")
    plt.show()
    
#96.8
#89.3
    cap.release()
    cv2.destroyAllWindows()
    print(move_mouse, frame_counter, left_click_count)
    print(move_mouse/frame_counter)
    print(left_click_count/frame_counter)
    print(right_click_count/frame_counter)

def sharpen():
    """ Function for creating hand_after_sharpening.jpg """
    img = cv2.imread("hand_before_sharpening.jpg")
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    cv2.imwrite("hand_after_sharpening.jpg", img)




def machine():
    """ Function for printing specs."""
    import sys
    print(sys.version)
    print(cv2.__version__)
    print(np.__version__)
    print(pyautogui.__version__)
    print(mp.__version__)

main()