import numpy as np
from tune_cropping import FindRatBoundingBox, get_background_avg
from collections import deque
from tqdm import tqdm
import cv2
import os

def crop_video(video_path, output_dir, num_frames_group, num_frames_crop, crop_parameters, desired_framerate=None):
    vid_name = os.path.basename(video_path).split('.')[0]
    out_sub_dir = os.path.join(output_dir, vid_name)

    bg_avg = get_background_avg(vid_path=video_path, num_frames=num_frames_crop)
    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    cropper = FindRatBoundingBox(bg_avg, **crop_parameters)

    frame_cache = deque(maxlen=num_frames_group)

    if fps:
        if desired_framerate is not None:
            if fps % desired_framerate == 0:
                skip_size = int(fps // desired_framerate)
            else:
                raise ValueError("Video framerate not an integer multiple of desired framerate")
        else:
            skip_size = 1
    else:
        raise ValueError("Cannot read framerate from video codec")
    
    frame_counter = 0
    processed_counter = 0
    pbar = tqdm(total=length // skip_size)
    
    while True:
        ret, frame = video.read()

        if not ret:
            break

        if frame_counter % skip_size != 0:
            frame_counter += 1
            continue
        
        *_, frame_cropped = cropper(frame)
        frame_cache.append(frame_cropped)

        # Save once cache is full, then for every subsequent frame
        if len(frame_cache) == num_frames_group:
            frame_folder = os.path.join(out_sub_dir, f'{processed_counter}')
            os.makedirs(frame_folder, exist_ok=True)

            for i, cached_frame in enumerate(frame_cache):
                out_path = os.path.join(frame_folder, f'{i}.png')
                cv2.imwrite(out_path, cached_frame)
        
        frame_counter += 1
        processed_counter += 1
        pbar.update(1)
    
    pbar.close()
    video.release()

def main():
    crop_parameters = {
        'box_size': 250,
        'difference_threshold': 10
    }

    desired_framerate = 30
    num_frames_group = 3
    num_frames_crop = 500

    vid_path = r"C:\Realtime\data\2026-01-09-11-05\01_09.mp4"
    out_path = r"C:\Users\Jimi\Desktop\Brayden\RealtimeLightning\RealtimeBehaviorClassification\processed_frames"

    crop_video(video_path=vid_path, output_dir=out_path, num_frames_group=num_frames_group, 
               desired_framerate=desired_framerate, num_frames_crop=num_frames_crop,
               crop_parameters=crop_parameters)

if __name__ == "__main__":
    main()