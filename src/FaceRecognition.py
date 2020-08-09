import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import face_recognition
import cv2

from VideoReader import VideoReader


class FaceRecognition():
    def __init__(self, knownFacesDir, videosDir):
        """
            @param knownFacesDir: (str) directory path which stores multiple images of known faces with person names as filename
            @param videosDir: (str) directory path of videos
        """
        self.persons = []           # List of person names
        self.faces = []             # List of face images. Each element is a 2D Numpy array
        self.facesEncodings = []    # List of face encodings. Each encoding has length of 128
        self._getKnownFaces(knownFacesDir)

        self.facesCount = [0] * len(self.persons)    # List of int
        self.videoReader = VideoReader(videosDir)    # new VideoReader
    

    def _getKnownFaces(self, knownFacesDir):
        """
        Init self attributes.
            @param knownFacesDir: (str) directory path which contains faces with their names as filename.
        """
        knownFacesFileList = glob(knownFacesDir + "/*")
        print(knownFacesFileList)
        for filename in knownFacesFileList:
            self.persons.append(filename.split('/')[-1].split('.')[0])
            img = face_recognition.load_image_file(filename)
            # H, W, Ch = img.shape
            # img = cv2.resize(img, (W//2, H//2))
            self.faces.append(img)
            self.facesEncodings.append(face_recognition.face_encodings(img)[0])


    def faceRecogn(self, image):
        """
        Get face id for a single input image.
            @param image: (Image) unknown image for recognition
            @return: (List) face index (or indices in case multiple faces)
        """
        res = []

        # Encode unknown face(s)
        unknownFaceEncodings = face_recognition.face_encodings(image)
        
        # Compare the input unknown face(s) with each known face
        for faceEncoding in unknownFaceEncodings:
            dists = face_recognition.face_distance(self.facesEncodings, faceEncoding)
            idx = np.argmin(dists)
            res.append(idx)
            self.facesCount[idx] += 1

        # Return true face index (or indices)
        return res

        
    def videoFaceRecogn(self):
        """
        For each frame in a video, recognize the face(s) and add to self.facesCount. Each frame is acquired by self.videoReader
        """
        for frame in tqdm(self.videoReader.readAllVideos()):      # generator
            _ = self.faceRecogn(frame)

