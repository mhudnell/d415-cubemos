# from skele_tracking import draw_beacon
from beacons import draw_beacon
import numpy as np
import time
import cv2


if __name__ == '__main__':

    # simulate camera running at 15 fps
    simulated_fps = 30
    animation_count = 0
    animation_length = 90
    while True:

        legal_image = np.ones((100,100,3), dtype=np.uint8) * 170
        violation_image = np.ones((100,100,3), dtype=np.uint8) * 170

        draw_beacon(legal_image, (50, 50), animation_count / animation_length, violation=False)
        draw_beacon(violation_image, (50, 50), animation_count / animation_length, violation=True)

        cv2.imshow("legal", legal_image)
        cv2.imshow("violation", violation_image)
        cv2.waitKey(int((1.0 / simulated_fps)*1000))

        animation_count += 1
        if animation_count == animation_length:
            animation_count = 0
