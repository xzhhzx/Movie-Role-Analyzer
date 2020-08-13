import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sys
from FaceRecognition import FaceRecognition


if __name__ == '__main__':
    personsDir = "../../persons"
    videosDir = "../../Video"
    # personsDir = sys.argv[1]
    # videosDir = sys.argv[2]

    fr = FaceRecognition(personsDir, videosDir)
    fr.videoFaceRecogn(numProcess=4, yieldNum=80, sampleRate=8)

    print(fr.persons)
    print(fr.facesCount)
    print(fr.videoReader.frameCount)
