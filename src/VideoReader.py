import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import cv2


class VideoReader():
    def __init__(self, videoFilesDir):
        """
            @param videoFilesDir: (str) directory path which stores all the videos
        """
        self.videoFilesDir = videoFilesDir
        self.videoFiles = glob(videoFilesDir + "/*")
        self.frameCount = []


    def readVideo(self, videoFile, sampleRate=25, resizeRate=2):
        """
        Read video and yield frame with a generator
        """
        video = cv2.VideoCapture(videoFile)
        self.frameCount.append(0)
        cnt = 0
        grabbed = True

        while(True): 
            if(cnt % sampleRate == 0):
                grabbed, frame = video.read()     # Read in frame
                H, W, Ch = frame.shape
                frame = cv2.resize(frame, (W//resizeRate, H//resizeRate))     # Resolution downsample
                if(grabbed): 
                    self.frameCount[-1] += 1    # Count total frames
                    yield frame
            else:       
                grabbed = video.grab()      # Grab frame but don't decode it and skip it
            
            cnt += 1   
            if(not grabbed):
                break            
            

    def readAllVideos(self):
        """
        Read all videos in self.videoFilesDir
        """
        for videoFile in self.videoFiles:
            print("Reading video file", videoFile, "......")
            for frame in self.readVideo(videoFile):
                yield frame