import pyrealsense2 as rs
import numpy as np
import cv2
import os
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
        (tuple(map(int, skeleton.joints[i])), tuple(map(int, skeleton.joints[v])))
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


def render_result(skeletons, img, confidence_threshold):
    skeleton_color = (100, 254, 213)
    for index, skeleton in enumerate(skeletons):
        limbs = get_valid_limbs(keypoint_ids, skeleton, confidence_threshold)
        for limb in limbs:
            cv2.line(
                img, limb[0], limb[1], skeleton_color, thickness=2, lineType=cv2.LINE_AA
            )

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
            render_result(skeletons, color_image, CONFIDENCE_THRESHOLD)

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
    finally:
        pipeline.stop()
