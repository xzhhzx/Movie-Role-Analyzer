import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import face_recognition
import cv2
import threading
import time
from multiprocessing import Pool, Process

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

        self.facesCount = np.array([0] * len(self.persons))    # List of int
        self.videoReader = VideoReader(videosDir)    # new VideoReader
    

    def _getKnownFaces(self, knownFacesDir):
        """
        Init self attributes.
            @param knownFacesDir: (str) directory path which contains faces with their names as filename.
        """
        knownFacesFileList = glob(knownFacesDir + "/*")
        for filename in knownFacesFileList:
            self.persons.append(filename.split('\\')[-1].split('.')[0])
            img = face_recognition.load_image_file(filename)
            # H, W, Ch = img.shape
            # img = cv2.resize(img, (W//2, H//2))
            self.faces.append(img)
            self.facesEncodings.append(face_recognition.face_encodings(img)[0])
        print(self.persons)


    def faceRecogn(self, image):
        """
        Get face appearances that are detected in a single input image.
            @param image: (Image) unknown image for recognition
            @return: (List) detected face of each person. E.g. res[i] == 1 if the i-th person appeared in the image
        """
        res = [0] * len(facesEncodings)
        start = time.time()

        # Encode unknown face(s)
        unknownFaceEncodings = face_recognition.face_encodings(image)
        
        # Compare the input unknown face(s) with each known face
        for faceEncoding in unknownFaceEncodings:
            dists = face_recognition.face_distance(self.facesEncodings, faceEncoding)
            idx = np.argmin(dists)
            res[idx] += 1
            self.facesCount[idx] += 1
        
        end = time.time()
        # print("Time of processing a single frame:", end-start)         # 1.5s (predominant part)

        # Return true face index (or indices)
        return res
    

    @staticmethod
    def batchedFacesRecogn(images, facesEncodings):
        """
        Get number of face appearances that are detected in multiple input images. This is a static method so that it can be invoked in child processes.
            @param image: (List<Image> or 3D/4D Numpy array) unknown image for recognition
            @return: (List) sum of detected faces in each image of each person. E.g. res[i] == 3 if the i-th person appeared 3 times in total
        """
        res = [0] * len(facesEncodings)
        start = time.time()

        # Iterate over each image
        for image in images:
            # Encode unknown face(s)
            unknownFaceEncodings = face_recognition.face_encodings(image)
            
            # Compare the input unknown face(s) with each known face
            for faceEncoding in unknownFaceEncodings:
                dists = face_recognition.face_distance(facesEncodings, faceEncoding)
                idx = np.argmin(dists)
                res[idx] += 1
            
        end = time.time()
        # print("Time of processing " + str(len(images)) + " frame:" + str(end-start))         # 1.5s (predominant part)

        # Return true face index (or indices)
        return res



    def videoFaceRecogn(self, numProcess=4, yieldNum=40, sampleRate=10):
        """
        For each frame in a video, recognize the face(s) and add to self.facesCount. Each frame is acquired by self.videoReader
            @param: numProcess: (int) 
            @param: yieldNum: (int)
            @param: sampleRate: (int)
        """
        s = time.time()
        assert(numProcess > 0)
        assert(yieldNum % numProcess == 0)      # Assert yieldNum can be perfectly divided into subtasks

        frameRate = 29.97   # unit: frame/s
        videoLength = 45    # unit: min
        print("Total approximate number of frames:", videoLength * 60 * frameRate // sampleRate)
        numFrames = 0
        ss = time.time()
        
        for frames in self.videoReader.readAllVideos(yieldNum=yieldNum, sampleRate=sampleRate):      # generator which generates 4 frames at a time
            
            print("Get", yieldNum, "frames")
            numFrames += len(frames)
            

            # Method 1.Sequential implementation
            if(numProcess == 1):
                for i in range(len(frames)):
                    _ = self.faceRecogn(frames[i])

            
            # Method 2.Multiprocess pool implementation
            else:
                # Data preperation

                frames = np.array(frames)   # [20, shape of frame]
                if(len(frames) == yieldNum):
                    frames = frames.reshape(numProcess, yieldNum//numProcess, *frames.shape[1:])    # [4, 5, shape of frame]
                else:
                    frames = frames.reshape(1, len(frames), *frames.shape[1:])      # Remainder, e.g. [1, 11, shape of frame]
                
                # Create pool
                with Pool(numProcess) as pool:   
                    args = zip(frames, [self.facesEncodings]*numProcess)    # Pack parameters  
                    resList = pool.starmap(self.batchedFacesRecogn, args)   # Distibute tasks to child processes

                for res in resList:     # Accumulate the returned results to master process
                    self.facesCount += res

            ee = time.time()
            print("Total time for processing", numFrames, "frames:", ee-ss, '\n')

        e = time.time()
        print("Total time:", e-s)



        # def process_tmp():
            
        #     while(True):
        #         cnt = 0
        #         while(self.tmp is None):
        #             print("Waiting for master to grab data...")
        #             cnt += 1
        #             # if(cnt > 1000): exit()
                
        #         if(self.tmp is -1):      # End
        #             break

        #         value = self.tmp
        #         self.tmp = None    # Reset tmp
        #         _ = self.faceRecogn(value)  # Time-consuming
            
        # x = threading.Thread(target=process_tmp, args=())
        # x.start()

        # for frame in tqdm(self.videoReader.readAllVideos()):      # generator
        #     while(self.tmp is not None):
        #         print("Waiting for worker to finish...")
                
        #     # print("In main grabbed frame")
        #     self.tmp = frame.copy()   # store frame
        
        # # Set tmp to -1 to signal worker thread to end
        # while(self.tmp is not None): 
        #     pass
        # self.tmp = -1
        # print("Set tmp to -1")

        # x.join()