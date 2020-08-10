import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import cv2
import time

class VideoReader():
    def __init__(self, videoFilesDir):
        """
            @param videoFilesDir: (str) directory path which stores all the videos
        """
        self.videoFilesDir = videoFilesDir
        self.videoFiles = glob(videoFilesDir + "/*")
        self.frameCount = []


    def readVideo(self, videoFile, sampleRate=3, resizeRate=2):
        """
        Generator: Read video and yield frame
        """
        video = cv2.VideoCapture(videoFile)
        self.frameCount.append(0)
        cnt = 0
        grabbed = True

        while(grabbed): 
            if(cnt % sampleRate == 0):
                grabbed, frame = video.read()     # Read in frame
                if(grabbed): 
                    H, W, Ch = frame.shape
                    frame = cv2.resize(frame, (W//resizeRate, H//resizeRate))     # Resolution downsample
                    self.frameCount[-1] += 1    # Count total frames
                    yield frame
            else:       
                grabbed = video.grab()      # Grab frame but don't decode it and skip it
            
            cnt += 1          
            

    def readAllVideos(self, yieldNum=20, sampleRate=3):
        """
        Generator: Read all videos in self.videoFilesDir
        """
        for videoFile in self.videoFiles:
            print("Reading video file", videoFile, "......")
            frames = []
            cnt = 0
            for frame in self.readVideo(videoFile, sampleRate=sampleRate):
                frames.append(frame)
                cnt += 1
                if(cnt == yieldNum):
                    yield frames
                    cnt = 0
                    frames = []
            yield frames        # Remaining frames < 4
                    

if __name__ == '__main__':
    vr = VideoReader("C:/Users/Zihan Xu/Desktop/videos")
    gen = vr.readVideo(vr.videoFiles[0])

    start = time.time() 
    for i in range(20):
        gen.__next__()
    end = time.time() 
    print("Average time of reading a frame:", (end-start) / 20)       # 0.0895 s