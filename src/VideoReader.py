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


    def readVideo(self, videoFile):
        """
        Read video and yield frame with a generator
        """
        video = cv2.VideoCapture(videoFile)
        self.frameCount.append(0)

        while(True): 
            ifGrabbed, frame = video.read()     # Read in
            if(ifGrabbed): 
                self.frameCount[-1] += 1    # Count total frames
                yield frame
            else: 
                break

    def readAllVideos(self):
        """
        Read all videos in self.videoFilesDir
        """
        for videoFile in self.videoFiles:
            print("Reading video file", videoFile, "......")
            for frame in self.readVideo(videoFile):
                yield frame