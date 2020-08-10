import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sys

import face_recognition
import cv2
from FaceRecognition import FaceRecognition

if __name__ == '__main__':
    personsDir = "C:/Users/Zihan Xu/Desktop/persons"
    videosDir = "C:/Users/Zihan Xu/Desktop/videos"
    # personsDir = sys.argv[1]
    # videosDir = sys.argv[2]

    fr = FaceRecognition(personsDir, videosDir)
    fr.videoFaceRecogn(numProcess=4, yieldNum=12, sampleRate=10)

    print(fr.persons)
    print(fr.facesCount)
    print(fr.videoReader.frameCount)