import pyrealsense2 as rs
import numpy as np
import cv2
import os
import math
from cubemos.core.nativewrapper import CM_TargetComputeDevice
from cubemos.core.nativewrapper import initialise_logging, CM_LogLevel
from cubemos.skeleton_tracking.nativewrapper import Api, SkeletonKeypoints
from cubemos_helpers import check_license_and_variables_exist, default_license_dir
from fps import FPS

keypoint_ids = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
]

def get_valid_limbs(keypoint_ids, skeleton, confidence_threshold):
    limbs = [
        (tuple(map(int, skeleton.joints[i])), tuple(map(int, skeleton.joints[v])), i, v)
        for (i, v) in keypoint_ids
        if skeleton.confidences[i] >= confidence_threshold
        and skeleton.confidences[v] >= confidence_threshold
    ]
    valid_limbs = [
        limb
        for limb in limbs
        if limb[0][0] >= 0 and limb[0][1] >= 0 and limb[1][0] >= 0 and limb[1][1] >= 0
    ]
    return valid_limbs

def draw_skeleton(img, skeleton, confidence_threshold, color):
    limbs = get_valid_limbs(keypoint_ids, skeleton, confidence_threshold)
    for limb in limbs:
        cv2.line(
            img, limb[0], limb[1], color, thickness=2, lineType=cv2.LINE_AA
        )
        # # draw keypoint index
        # cv2.putText(
        #     img, str(limb[2]), limb[0], cv2.FONT_HERSHEY_COMPLEX, fontScale=0.7, color=(0, 255, 0), thickness=2
        # )
        # cv2.putText(
        #     img, str(limb[3]), limb[1], cv2.FONT_HERSHEY_COMPLEX, fontScale=0.7, color=(0, 255, 0), thickness=2
        # )


def draw_beacon(img, pt, animation_percentage, violation=False):
    # determine alpha
    animation_alpha_start = 0
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
        bgra = (0, 0 , 255, alpha)
    else:
        bgra = (0, 255, 0, alpha)

    # draw circle in buffer
    overlay = np.copy(img)
    cv2.circle(overlay, (int(pt[0]), int(pt[1])), 12, bgra, thickness=cv2.FILLED)

    # blend
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)

def render_result(skeletons, img, confidence_threshold, depth_frame, depth_scale, animation_percentage):
    skeleton_color = (100, 254, 213)
    skeles_drawn = np.zeros(len(skeletons))
    for i, skeleton in enumerate(skeletons):
        if skeles_drawn[i]: # skip if already drawn
            continue

        # draw_skeleton(img, skeleton, confidence_threshold, skeleton_color)
        # determine distance violations - mh
        chest1_keypoint = skeleton.joints[1]
        if (chest1_keypoint[0] < 0 or chest1_keypoint[0] >= 640) or (chest1_keypoint[1] < 0 or chest1_keypoint[1] >= 480):
            break
        
        # chest1_distance = depth_frame.get_distance(int(chest1_keypoint[0]), int(chest1_keypoint[1]))
        depth = depth_image[int(chest1_keypoint[1]), int(chest1_keypoint[0])] * depth_scale
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        p1 = rs.rs2_deproject_pixel_to_point(depth_intrin, [int(chest1_keypoint[1]), int(chest1_keypoint[0])], depth)
        # p1 = [chest1_keypoint[0], chest1_keypoint[1], chest1_distance]

        for j in range(i+1, len(skeletons)):
            skeleton2 = skeletons[j]
            chest2_keypoint = skeleton2.joints[1]
            if (chest2_keypoint[0] < 0 or chest2_keypoint[0] >= 640) or (chest2_keypoint[1] < 0 or chest2_keypoint[1] >= 480):
                break
            # chest2_distance = depth_frame.get_distance(int(chest2_keypoint[0]), int(chest2_keypoint[1]))
            depth = depth_image[int(chest2_keypoint[1]), int(chest2_keypoint[0])] * depth_scale
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            p2 = rs.rs2_deproject_pixel_to_point(depth_intrin, [int(chest2_keypoint[1]), int(chest2_keypoint[0])], depth)
            # p2 = [chest2_keypoint[0], chest2_keypoint[1], chest1_distance]

            distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) +((p1[2]-p2[2])**2))

            if distance < 1.5:  # draw skele red
                # draw_skeleton(img, skeleton, confidence_threshold, (0, 0 , 255))
                # draw_skeleton(img, skeleton2, confidence_threshold, (0, 0 , 255))

                draw_beacon(img, chest1_keypoint, animation_percentage, violation=True)
                draw_beacon(img, chest2_keypoint, animation_percentage, violation=True)

                skeles_drawn[i] = 1
                skeles_drawn[j] = 1
                break

        if not skeles_drawn[i]:  # draw skele green
            # draw_skeleton(img, skeleton, confidence_threshold, skeleton_color)

            draw_beacon(img, chest1_keypoint, animation_percentage, violation=False)



if __name__ == '__main__':

    # Configure depth and color streams
    image_width = 640
    image_height = 480
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, image_width, image_height, rs.format.bgr8, 30)

    # get skeleton tracking api
    sdk_path = os.environ["CUBEMOS_SKEL_SDK"]
    api = Api(default_license_dir())
    model_path = os.path.join(
        sdk_path, "models", "skeleton-tracking", "fp32", "skeleton-tracking.cubemos"
    )
    api.load_model(CM_TargetComputeDevice.CM_CPU, model_path)
    CONFIDENCE_THRESHOLD = 0.5

    # Start streaming
    profile = pipeline.start(config)

    # Create alignment primitive with color as its target stream:
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    # initialize fps counting
    frame_count = 0 # counts every 15 frames
    fps_update_rate = 15 # 1 # the interval (of frames) at which the fps is updated
    fps_deque_size = 2 # 5
    fps = FPS(deque_size=fps_deque_size, update_rate=fps_update_rate)
    curr_fps = 0

    # animation state variables
    animation_count = 0
    animation_length = 90

    try:
        while True:

            ###### Wait for a coherent pair of frames, and align them
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not depth_frame or not color_frame: # try until both images are ready
                continue
                
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # perform inference
            skeletons = api.estimate_keypoints(color_image, 192)
            render_result(skeletons, color_image, CONFIDENCE_THRESHOLD, depth_frame, depth_scale, animation_count / animation_length)

            # compute fps
            if frame_count == fps_update_rate:
                frame_count = 0
                fps.update()
                curr_fps = fps.fps()

            # draw fps on image
            cv2.putText(
                color_image, str(int(curr_fps)),
                (0, 15), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 0), thickness=2
            )
            frame_count += 1

            ###### Display frame, exit if key press
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)
            k = cv2.waitKey(1)
            if k != -1:  # exit if key pressed   ESC (k=27)
                cv2.destroyAllWindows()
                break

            # increment animation
            animation_count += 1
            if animation_count == animation_length:
                animation_count = 0
    finally:
        pipeline.stop()
