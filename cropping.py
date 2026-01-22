import cv2
import numpy as np
from functools import partial

class FindMouseBoundingBox():
    def __init__(self, bg, difference_threshold=5, padding_ratio=1.4, min_box_sidelen=20, downsample_factor=1,
                 open1_kernel=4, close_kernel=15, open2_kernel=10):
        self.difference_threshold = difference_threshold
        self.padding_ratio = padding_ratio
        self.min_box_sidelen = min_box_sidelen
        self.downsample_factor = downsample_factor

        self.h, self.w = bg.shape[:2]
        self.resizer = partial(cv2.resize, dsize=(self.w // downsample_factor, self.h // downsample_factor), interpolation=cv2.INTER_NEAREST)
        self.bg = self.resizer(bg)
        self.prev_bounding = (0, 0, self.w, self.h)

        self.open1_kernel = open1_kernel
        self.close_kernel = close_kernel
        self.open2_kernel = open2_kernel
    
    def resize(self, frame):
        return self.resizer(frame)

    def crop(self, frame):
        small = self.resizer(frame)

        thresholded = ((np.abs(small - self.bg) > self.difference_threshold)*255).astype(np.uint8)

        opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, np.ones((self.open1_kernel, self.open1_kernel), np.uint8))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((self.close_kernel, self.close_kernel), np.uint8))
        opened_again = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((self.open2_kernel, self.open2_kernel), np.uint8))

        contours, _ = cv2.findContours(opened_again, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)

            # area = cv2.contourArea(cnt)

            # if area < 3000:
            (x, y, w, h) = cv2.boundingRect(cnt)

            side_length = max((w, h, self.min_box_sidelen))
            padded_side = int(side_length * self.padding_ratio)
            center_x, center_y = x + w // 2, y + h // 2
            new_x = max(0, center_x - padded_side // 2)
            new_y = max(0, center_y - padded_side // 2)

            bb = (new_x, new_y, padded_side, padded_side)

            self.prev_bounding = bb

            return bb, small, opened_again
            
        print("No contours detected")
        return self.prev_bounding, small, opened_again
    
    def __call__(self, frame):
        return self.crop(frame)

def get_background_avg(vid_path, num_frames):
    cap = cv2.VideoCapture(vid_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = np.zeros((num_frames, height, width))

    idxs = np.random.randint(0, frame_count, num_frames)

    for i, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            raise Exception("Can't read frame")

        frames[i] = frame[..., 0]
    
    avg = frames.mean(axis=0)

    cap.release()

    return avg

def find_rat_bounding_box(frame, bg, difference_threshold, padding_ratio, min_box_sidelen, downsample_factor=2):
    h, w = frame.shape[:2]
    # small_frame = cv2.resize(frame, (w // downsample_factor, h // downsample_factor), interpolation=cv2.INTER_NEAREST)

    small = ((np.abs(small - bg) > difference_threshold)*255).astype(np.uint8)

    small_dilated = cv2.morphologyEx(small, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

    small_closed = cv2.morphologyEx(small_dilated, cv2.MORPH_CLOSE, np.ones())

    small_dilated_again = cv2.morphologyEx(small_closed, cv2.MORPH_OPEN, np.ones((10, 10), np.uint8))

    contours, _ = cv2.findContours(small_dilated_again, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)

        (x, y, w, h) = cv2.boundingRect(cnt)

        side_length = max((w, h, min_box_sidelen))
        padded_side = int(side_length * padding_ratio)
        center_x, center_y = x + w // 2, y + h // 2
        new_x = max(0, center_x - padded_side // 2)
        new_y = max(0, center_y - padded_side // 2)

def main():
    video_path = r"C:\Users\Jimi\Desktop\Brayden\Realtime_Model\data\Rig1_10_29.mp4"

    num_frames = 1000
    background_avg = get_background_avg(video_path, num_frames)

    cap = cv2.VideoCapture(video_path)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    difference_threshold = 15
    padding = 1.6
    min_box_sidelen = 20
    resize_scale = 1
    kernel = np.ones((100, 100), np.uint8)

    background_avg_small = cv2.resize(background_avg, None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_NEAREST)
    cropper = FindRatBoundingBox(background_avg, difference_threshold=difference_threshold, padding_ratio=padding, 
                                 min_box_sidelen=min_box_sidelen, downsample_factor=1)

    new_x, new_y, padded_side = 0, 0, 500

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = frame[..., 0]
        small = cv2.resize(frame.copy(), None, fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_NEAREST)

        small_duped = small.copy()

        small_thresholded = ((np.abs(small_duped - background_avg_small) > difference_threshold)*255).astype(np.uint8)

        small_dilated = cv2.morphologyEx(small_thresholded, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

        small_closed = cv2.morphologyEx(small_dilated, cv2.MORPH_CLOSE, kernel)

        small_dilated_again = cv2.morphologyEx(small_closed, cv2.MORPH_OPEN, np.ones((10, 10), np.uint8))

        contours, _ = cv2.findContours(small_dilated_again, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)

            area = cv2.contourArea(cnt)

            if area < 2000:
                (x, y, w, h) = cv2.boundingRect(cnt)

                side_length = max((w, h, min_box_sidelen))
                padded_side = int(side_length * padding)
                center_x, center_y = x + w // 2, y + h // 2
                new_x = max(0, center_x - padded_side // 2)
                new_y = max(0, center_y - padded_side // 2)

            # Draw rectangle on the frame (blue box)
            cv2.rectangle(small_dilated_again, (new_x, new_y), (new_x + padded_side, new_y + padded_side), (255, 255, 255), 2)
        
        cropped_box = small_duped[new_y:new_y+padded_side, new_x:new_x+padded_side]
        (x, y, w, h), small = cropper(frame)
        cropped_obj = small[y:y+w, x:x+w]

        cv2.imshow(f'Background Subtracted', cv2.resize(small_dilated, (500, 500)))
        cv2.imshow("Original", cv2.resize(small_dilated_again, (500, 500)))
        cv2.imshow("Full Processed", cv2.resize(small_closed, (500, 500)))
        cv2.imshow("Cropped Small", cv2.resize(cropped_box, (500, 500)))
        cv2.imshow("Cropper Object", cv2.resize(cropped_obj, (500, 500)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    