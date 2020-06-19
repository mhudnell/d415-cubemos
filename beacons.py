import cv2
import numpy as np

def draw_beacon(img, pt, animation_percentage, violation=False):
    # determine alpha
    animation_alpha_start = 100
    animation_alpha_end = 255
    violation_blinks_per_animation = 3

    if violation:
        if int(animation_percentage * violation_blinks_per_animation * 2) % 2 == 0:
            alpha = 0
        else:
            alpha = 1.0
    else:
        if animation_percentage < 0.5:  # increment in first half of animation
            alpha = (animation_alpha_start + (animation_alpha_end - animation_alpha_start)*animation_percentage*2) / 255
        else:  # decrement in second half of animation
            alpha = (animation_alpha_end - (animation_alpha_end - animation_alpha_start)*(animation_percentage-.5)*2) / 255

    # set color
    if violation:
        # bgra = (11, 43, 222, int(255*alpha))
        bgr = (0, 0, 242)
    else:
        # bgra = (0, 255, 0, int(255*alpha))
        bgr = (0, 250, 0)


    # draw circle in buffer
    overlay = np.copy(img)
    # cv2.circle(overlay, (int(pt[0]), int(pt[1])), 12, bgra, thickness=cv2.FILLED)
    cv2.circle(overlay, (int(pt[0]), int(pt[1])), 12, bgr, thickness=4, lineType=cv2.LINE_AA)

    # blend
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)

# def draw_beacon()