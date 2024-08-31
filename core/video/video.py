import time
from threading import Thread
from typing import Union

import cv2

class VideoStream:
    def __init__(self, video_source: Union[int, str], WEBCAM_HFLIP=False, WEBCAM_VFLIP=False, 
                 CAM_WIDTH=None, CAM_HEIGHT=None):
        """
        initialize the video camera stream and read the first frame from the stream
        
        Args:
            video_source (int): The ID of the video source to use.
            WEBCAM_HFLIP (bool): Whether to flip the video horizontally.
            WEBCAM_VFLIP (bool): Whether to flip the video vertically.
            CAM_WIDTH (int): The width of the video to capture.
            CAM_HEIGHT (int): The height of the video to capture.
        """
        self.WEBCAM_HFLIP, self.WEBCAM_VFLIP = WEBCAM_HFLIP, WEBCAM_VFLIP
        self.stream = cv2.VideoCapture(video_source)
        self.video_source = video_source
        
        # if self.isWebCam():
        #     self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        #     self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        
        self.start_time = time.time()
        self.frames_read = 0
        self.thread = None
        
        # initialize the variable used to indicate if the thread should be stopped
        self.stopped = False
        
        # add fps tracking
        self.proc_fps_info, self.proc_fps = { "prev_frames_read":0, "fps_time": time.time() }, 0

    def start(self):
        """start the thread to read frames from the video stream"""
        if type(self.video_source) == int:
            self.thread = Thread(target=self.update, args=())
            self.thread.daemon = True
            
            max_tries = 3
            while max_tries > 0:
                self.thread.start()
                time.sleep(4.0) 
                if self.read() is not None: break
                max_tries -= 1
        else:
            self.update()
            self.update()
        
        # Get video sizes 
        for i in range(1): self.read() # Fix bug where first image is read with incorrect size
        assert self.frame is not None, "Error loading video source. No frames recieved from source."
        
        self.height = self.frame.shape[0]
        self.width = self.frame.shape[1]

    def offset_seeker_ms(self, offset_ms):
        """Offset clip streaming by ms +/-. Note: restarts stream to offset from beginning."""
        raise NotImplementedError("Offset stopped working, unclear why. Please trim video manually.")
        assert type(self.video_source) == str, "Offset only works with files."
        self.stream.set(cv2.CAP_PROP_POS_MSEC, offset_ms)
        _, _ = self.stream.read()
        
    def get_time_ms(self):
        """ Get current time in clip in ms. """
        if type(self.video_source) == str: return self.stream.get(cv2.CAP_PROP_POS_MSEC) 
        else: return time.time() - self.start_time
        
    def update_proc_fps(self):
        """Calculate and display frames per second processing"""
        if self.frames_read >= 60:
            duration = float(time.time() - self.proc_fps_info['fps_time'])
            elapsed_frames = self.frames_read - self.proc_fps_info['prev_frames_read']
            self.proc_fps = float(elapsed_frames / duration)
            self.proc_fps_info = { "prev_frames_read":self.frames_read, "fps_time": time.time() }

    def update(self):
        """keep looping infinitely until the thread is stopped"""
        if type(self.video_source) == int:
            while True:
                # if the thread indicator variable is set, stop the thread
                if self.stopped:
                    self.stream.release()
                    return
                # otherwise, read the next frame from the stream
                (self.grabbed, self.frame) = self.stream.read()
                # check for valid frames
                if not self.grabbed:
                    # no frames recieved, then safely exit
                    self.stopped = True
                self.frames_read += 1
            
        else:
            (self.grabbed, self.frame) = self.stream.read()
            # check for valid frames
            if not self.grabbed:
                # no frames recieved, then safely exit
                self.stopped = True
                self.frame = None
            else:
                self.frames_read += 1
        
    def read(self):
        """return the frame most recently read
        Note there will be a significant performance hit to
        flip the webcam image so it is advised to just
        physically flip the camera and avoid
        setting WEBCAM_HFLIP = True or WEBCAM_VFLIP = True
        """
        if type(self.video_source) == str:
            self.update()
        
        if self.WEBCAM_HFLIP and self.WEBCAM_VFLIP:
            self.frame = cv2.flip(self.frame, -1)
        elif self.WEBCAM_HFLIP:
            self.frame = cv2.flip(self.frame, 1)
        elif self.WEBCAM_VFLIP:
            self.frame = cv2.flip(self.frame, 0)
        
        if self.frames_read-self.proc_fps_info['prev_frames_read'] > 60: self.update_proc_fps()
        
        return self.frame, self.frames_read

    def stop(self):
        """indicate that the thread should be stopped"""
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        if self.thread is not None:
            self.thread.join()  # properly handle thread exit

    def isOpened(self):
        return self.stream.isOpened()

    def isWebCam(self):
        return type(self.video_source) == int
