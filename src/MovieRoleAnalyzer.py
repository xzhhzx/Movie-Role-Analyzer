import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sys
from FaceRecognition import FaceRecognition


if __name__ == '__main__':
    personsDir = "../../persons"
    videosDir = "../../videos"
    # personsDir = sys.argv[1]
    # videosDir = sys.argv[2]

    for k in [1,2,3]:
        fr = FaceRecognition(personsDir, videosDir)
        fr.videoFaceRecogn(numProcess=4, yieldNum=40, sampleRate=8, k_certainty = k)

    # print(fr.persons)
    # print(fr.facesCount)
    # print(fr.videoReader.frameCount)
