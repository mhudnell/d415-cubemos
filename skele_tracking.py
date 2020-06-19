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
from beacons import draw_beacon

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

def get_midpoint(coord1, coord2):
    """
    expects Cubemos Coordinate as input
    returns tuple of 2 ints
    """
    return (int((coord1[0] + coord2[0]) // 2), int((coord1[1] + coord2[1]) // 2))

def render_result(skeletons, img, confidence_threshold, depth_frame, depth_scale, animation_percentage):
    skeleton_color = (100, 254, 213)
    skeles_drawn = np.zeros(len(skeletons))
    for i, skeleton in enumerate(skeletons):
        if skeles_drawn[i] or skeleton.confidences[1] < confidence_threshold: # skip if already drawn or if chest confidence is too low
            continue

        # draw_skeleton(img, skeleton, confidence_threshold, skeleton_color)
        # determine distance violations - mh
        chest1_keypoint = skeleton.joints[1]
        # print(chest1_keypoint)
        # print(skeleton.joints[8])
        # print(get_midpoint(chest1_keypoint, skeleton.joints[8]))
        pelvis_pt1 = get_midpoint(skeleton.joints[8], skeleton.joints[11])
        draw_pt1 = get_midpoint(chest1_keypoint, pelvis_pt1)
        if (chest1_keypoint[0] < 0 or chest1_keypoint[0] >= 640) or (chest1_keypoint[1] < 0 or chest1_keypoint[1] >= 480):
            break
        
        # chest1_distance = depth_frame.get_distance(int(chest1_keypoint[0]), int(chest1_keypoint[1]))
        depth = depth_image[int(chest1_keypoint[1]), int(chest1_keypoint[0])] * depth_scale
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        p1 = rs.rs2_deproject_pixel_to_point(depth_intrin, [int(chest1_keypoint[1]), int(chest1_keypoint[0])], depth)
        # p1 = [chest1_keypoint[0], chest1_keypoint[1], chest1_distance]

        for j in range(i+1, len(skeletons)):
            skeleton2 = skeletons[j]

            # skip if chest confidence is too low
            if skeleton2.confidences[1] < confidence_threshold:
                continue

            chest2_keypoint = skeleton2.joints[1]
            pelvis_pt2 = get_midpoint(skeleton2.joints[8], skeleton2.joints[11])
            draw_pt2 = get_midpoint(chest2_keypoint, pelvis_pt2)
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


                draw_beacon(img, draw_pt1, animation_percentage, violation=True)
                draw_beacon(img, draw_pt2, animation_percentage, violation=True)

                skeles_drawn[i] = 1
                skeles_drawn[j] = 1
                break

        if not skeles_drawn[i]:  # draw skele green
            # draw_skeleton(img, skeleton, confidence_threshold, skeleton_color)

            draw_beacon(img, draw_pt1, animation_percentage, violation=False)



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
