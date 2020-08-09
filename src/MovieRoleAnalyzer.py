import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import face_recognition
import cv2

from FaceRecognition import FaceRecognition


fr = FaceRecognition(sys.argv[1], sys.argv[2])
fr.videoFaceRecogn()