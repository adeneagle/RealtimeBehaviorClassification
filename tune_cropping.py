import cv2
import numpy as np
from functools import partial

class FindRatBoundingBox():
    def __init__(self, bg, difference_threshold=5, box_size=100, 
                 downsample_factor=1, open1_kernel=4, close_kernel=15, open2_kernel=10):
        self.difference_threshold = difference_threshold
        self.box_size = box_size
        self.downsample_factor = downsample_factor
        self.open1_kernel = open1_kernel
        self.close_kernel = close_kernel
        self.open2_kernel = open2_kernel
        self.h, self.w = bg.shape[:2]
        self.resizer = partial(cv2.resize, dsize=(self.w // downsample_factor, self.h // downsample_factor), interpolation=cv2.INTER_NEAREST)
        self.bg = self.resizer(bg)
        self.prev_bounding = (0, 0, self.w, self.h)
    
    def resize(self, frame):
        return self.resizer(frame)
    
    def crop(self, frame):
        # Convert RGB to grayscale for processing if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        small = self.resizer(gray)
        thresholded = ((np.abs(small - self.bg) > self.difference_threshold)*255).astype(np.uint8)
        opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, 
                                  np.ones((self.open1_kernel, self.open1_kernel), np.uint8))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, 
                                  np.ones((self.close_kernel, self.close_kernel), np.uint8))
        opened_again = cv2.morphologyEx(closed, cv2.MORPH_OPEN, 
                                        np.ones((self.open2_kernel, self.open2_kernel), np.uint8))
        contours, _ = cv2.findContours(opened_again, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            # Calculate center of mass using moments
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
            else:
                # Fallback to bounding box center if moments fail
                (x, y, w, h) = cv2.boundingRect(cnt)
                center_x = x + w // 2
                center_y = y + h // 2
            
            # Create fixed-size box centered on center of mass
            new_x = max(0, center_x - self.box_size // 2)
            new_y = max(0, center_y - self.box_size // 2)
            bb = (new_x, new_y, self.box_size, self.box_size)
            self.prev_bounding = bb
            
            # Crop from original frame (scale back to original size)
            x_scaled = new_x * self.downsample_factor
            y_scaled = new_y * self.downsample_factor
            w_scaled = self.box_size * self.downsample_factor
            h_scaled = self.box_size * self.downsample_factor
            
            # Ensure crop doesn't go out of bounds
            x_scaled = max(0, min(x_scaled, frame.shape[1] - w_scaled))
            y_scaled = max(0, min(y_scaled, frame.shape[0] - h_scaled))
            x_end = min(x_scaled + w_scaled, frame.shape[1])
            y_end = min(y_scaled + h_scaled, frame.shape[0])
            
            cropped_rgb = frame[y_scaled:y_end, x_scaled:x_end]
            
            return bb, small, opened_again, cropped_rgb
        
        print("No contours detected")
        # Return previous bounding box and empty crop
        x_scaled = self.prev_bounding[0] * self.downsample_factor
        y_scaled = self.prev_bounding[1] * self.downsample_factor
        w_scaled = self.prev_bounding[2] * self.downsample_factor
        h_scaled = self.prev_bounding[3] * self.downsample_factor
        
        x_scaled = max(0, min(x_scaled, frame.shape[1] - w_scaled))
        y_scaled = max(0, min(y_scaled, frame.shape[0] - h_scaled))
        x_end = min(x_scaled + w_scaled, frame.shape[1])
        y_end = min(y_scaled + h_scaled, frame.shape[0])
        
        cropped_rgb = frame[y_scaled:y_end, x_scaled:x_end]
        
        return self.prev_bounding, small, opened_again, cropped_rgb
    
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
        frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg = frames.mean(axis=0)
    cap.release()
    return avg

class ParameterTuner:
    def __init__(self, video_path, num_bg_frames=500):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initial parameters
        self.params = {
            'difference_threshold': 5,
            'box_size': 100,
            'downsample_factor': 1,
            'open1_kernel': 4,
            'close_kernel': 15,
            'open2_kernel': 10
        }
        
        # Get background
        print("Computing background...")
        self.bg = get_background_avg(video_path, num_bg_frames)
        
        # Create window and trackbars
        cv2.namedWindow('Parameter Tuner', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Parameter Tuner', 1200, 800)
        cv2.moveWindow('Parameter Tuner', 50, 50)
        
        cv2.createTrackbar('Diff Threshold', 'Parameter Tuner', 
                          self.params['difference_threshold'], 50, self.update_diff_threshold)
        cv2.createTrackbar('Box Size', 'Parameter Tuner', 
                          self.params['box_size'], 500, self.update_box_size)
        cv2.createTrackbar('Downsample', 'Parameter Tuner', 
                          self.params['downsample_factor'], 4, self.update_downsample)
        cv2.createTrackbar('Open1 Kernel', 'Parameter Tuner', 
                          self.params['open1_kernel'], 30, self.update_open1_kernel)
        cv2.createTrackbar('Close Kernel', 'Parameter Tuner', 
                          self.params['close_kernel'], 30, self.update_close_kernel)
        cv2.createTrackbar('Open2 Kernel', 'Parameter Tuner', 
                          self.params['open2_kernel'], 30, self.update_open2_kernel)
        
        # Playback control
        self.playing = True
        cv2.createTrackbar('Play/Pause (0=pause)', 'Parameter Tuner', 1, 1, self.toggle_play)
        
        # Initialize tracker
        self.update_tracker()
    
    def update_diff_threshold(self, val):
        self.params['difference_threshold'] = max(1, val)
        self.update_tracker()
    
    def update_box_size(self, val):
        self.params['box_size'] = max(10, val)
        self.update_tracker()
    
    def update_downsample(self, val):
        self.params['downsample_factor'] = max(1, val)
        self.update_tracker()
    
    def update_open1_kernel(self, val):
        self.params['open1_kernel'] = max(1, val)
        self.update_tracker()
    
    def update_close_kernel(self, val):
        self.params['close_kernel'] = max(1, val)
        self.update_tracker()
    
    def update_open2_kernel(self, val):
        self.params['open2_kernel'] = max(1, val)
        self.update_tracker()
    
    def toggle_play(self, val):
        self.playing = (val == 1)
    
    def update_tracker(self):
        self.tracker = FindRatBoundingBox(
            self.bg,
            difference_threshold=self.params['difference_threshold'],
            box_size=self.params['box_size'],
            downsample_factor=self.params['downsample_factor'],
            open1_kernel=self.params['open1_kernel'],
            close_kernel=self.params['close_kernel'],
            open2_kernel=self.params['open2_kernel']
        )
    
    def run(self):
        print("\nControls:")
        print("  SPACE: Play/Pause")
        print("  R: Restart video")
        print("  Q/ESC: Quit")
        print("  S: Save current parameters")
        print("\nAdjust trackbars to tune parameters in real-time\n")
        
        while True:
            if self.playing:
                ret, frame = self.cap.read()
                if not ret:
                    # Loop video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        break
            else:
                # Stay on current frame
                current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                if current_pos > 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
                ret, frame = self.cap.read()
                if not ret:
                    break
            
            # Convert to grayscale
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Get bounding box and processed frame
            (x, y, w, h), small, mask, cropped_rgb = self.tracker.crop(frame)
            
            # Scale bounding box back to original size
            scale = self.params['downsample_factor']
            x_orig = x * scale
            y_orig = y * scale
            w_orig = w * scale
            h_orig = h * scale
            
            # Draw bounding box on original frame
            frame_with_box = frame.copy()
            cv2.rectangle(frame_with_box, (x_orig, y_orig), 
                         (x_orig + w_orig, y_orig + h_orig), (0, 255, 0), 2)
            
            # Use the cropped RGB from tracker
            crop_size = 300
            if cropped_rgb.shape[0] > 0 and cropped_rgb.shape[1] > 0:
                cropped_display = cv2.resize(cropped_rgb, (crop_size, crop_size))
            else:
                cropped_display = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            
            # Create display layout
            # Resize for display if needed
            display_width = 400
            scale_factor = display_width / self.width
            display_height = int(self.height * scale_factor)
            
            frame_display = cv2.resize(frame_with_box, (display_width, display_height))
            
            # Convert mask to 3-channel for display
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_display = cv2.resize(mask_colored, (display_width, display_height))
            
            # Add text labels
            cv2.putText(frame_display, "Original + BBox", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(mask_display, "Processed Mask", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(cropped_display, "Cropped", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Combine displays side by side
            top_row = np.hstack([frame_display, mask_display])
            
            # Add cropped to bottom (centered)
            bottom_padding = (top_row.shape[1] - crop_size) // 2
            bottom_row = np.zeros((crop_size, top_row.shape[1], 3), dtype=np.uint8)
            bottom_row[:, bottom_padding:bottom_padding+crop_size] = cropped_display
            
            # Add parameter text
            param_display = np.zeros((150, top_row.shape[1], 3), dtype=np.uint8)
            y_pos = 25
            cv2.putText(param_display, f"Diff Threshold: {self.params['difference_threshold']}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 25
            cv2.putText(param_display, f"Box Size: {self.params['box_size']}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 25
            cv2.putText(param_display, f"Downsample: {self.params['downsample_factor']}", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            y_pos = 25
            cv2.putText(param_display, f"Open1 Kernel: {self.params['open1_kernel']}x{self.params['open1_kernel']}", 
                       (400, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 25
            cv2.putText(param_display, f"Close Kernel: {self.params['close_kernel']}x{self.params['close_kernel']}", 
                       (400, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 25
            cv2.putText(param_display, f"Open2 Kernel: {self.params['open2_kernel']}x{self.params['open2_kernel']}", 
                       (400, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Stack everything
            display = np.vstack([top_row, bottom_row, param_display])
            
            cv2.imshow('Parameter Tuner', display)
            
            # Handle key presses
            key = cv2.waitKey(int(1000/self.fps) if self.playing else 1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # SPACE
                self.playing = not self.playing
                cv2.setTrackbarPos('Play/Pause (0=pause)', 'Parameter Tuner', 1 if self.playing else 0)
            elif key == ord('r'):  # R
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            elif key == ord('s'):  # S
                self.save_parameters()
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def save_parameters(self):
        print("\n" + "="*60)
        print("CURRENT PARAMETERS:")
        print("="*60)
        print(f"FindRatBoundingBox(")
        print(f"    bg,")
        print(f"    difference_threshold={self.params['difference_threshold']},")
        print(f"    box_size={self.params['box_size']},")
        print(f"    downsample_factor={self.params['downsample_factor']},")
        print(f"    open1_kernel={self.params['open1_kernel']},")
        print(f"    close_kernel={self.params['close_kernel']},")
        print(f"    open2_kernel={self.params['open2_kernel']}")
        print(f")")
        print("="*60 + "\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    tuner = ParameterTuner(video_path)
    tuner.run()