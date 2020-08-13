import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import face_recognition
import cv2
import dlib
import threading
import time
import re
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
        self.stat = {}       # {name: faceCount} 
    

    def _getKnownFaces(self, knownFacesDir):
        """
        Init self attributes.
            @param knownFacesDir: (str) directory path which contains faces with their names as filename.
        """
        knownFacesFileList = glob(knownFacesDir + "/*")
        for filename in knownFacesFileList:
            if('\\' in filename):
                self.persons.append(filename.split('\\')[-1].split('.')[0])     # Windows format
            else:
                self.persons.append(filename.split('/')[-1].split('.')[0])      # Linux format
            img = face_recognition.load_image_file(filename)
            # H, W, Ch = img.shape
            # img = cv2.resize(img, (W//2, H//2))
            self.faces.append(img)
            self.facesEncodings.append(face_recognition.face_encodings(img)[0])
        # print("_getKnownFaces:", self.persons)


    @staticmethod
    def _faceRecogn(image, facesEncodings, personNames, k=3):
        """
        Detect faces in image with comparision to facesEncodings. Save result in res
            @param image: (Image) unknown image for recognition
            @param facesEncodings: (List<Vector>) known faces encodings 
            @param personNames: (List<str>)
            @param k: (int) number of minimum 
            @return: (List) detected face of each person. E.g. res[i] == 1 if the i-th person appeared in the image
        """
        res = [0] * len(facesEncodings)

        # Encode unknown face(s)
        unknownFaceEncodings = face_recognition.face_encodings(image)
        
        # Compare the input unknown face(s) with each known face
        for faceEncoding in unknownFaceEncodings:
            dists = face_recognition.face_distance(facesEncodings, faceEncoding)
            # idx = np.argmin(dists)
            minIndices = np.argpartition(dists, k)[:k]
            
            names = []
            for idx in minIndices:
                name = personNames[idx]
                name = name[:len(name) -1 - len(name.split("_")[-1])]      # Get name 
                names.append(name)

            # If minimum k distances does belong to the same person, accumulate
            # Or else it might be a Long Tao who is not in self.persons
            if(names.count(names[0]) == len(names)):
                res[minIndices[0]] += 1

        return res


    def faceRecogn(self, image, k=3):
        """
        Get face appearances that are detected in a single input image.
            @param image: (Image) unknown image for recognition
            @return: (List) detected face of each person. E.g. res[i] == 1 if the i-th person appeared in the image
        """
        res = self._faceRecogn(image, self.facesEncodings, self.persons, k)
        self.facesCount += res      # Accumulate to faceCount
        return res
    

    @staticmethod
    def batchedFacesRecogn(images, facesEncodings, persons, k=3):
        """
        Get number of face appearances that are detected in multiple input images. This is a static method so that it can be invoked in child processes.
            @param image: (List<Image> or 3D/4D Numpy array) unknown image for recognition
            @oaram facesEncodings: (List<Vector>) known faces encodings 
            @return: (1D Numpy array) sum of detected faces in each image of each person. E.g. res[i] == 3 if the i-th person appeared 3 times in total
        """
        res = np.array([0] * len(facesEncodings))

        # Iterate over each image
        for image in images:
            res += FaceRecognition._faceRecogn(image, facesEncodings, persons, k)
        return res    



    def videoFaceRecogn(self, numProcess=4, yieldNum=40, sampleRate=10, k_certainty=3):
        """
        For each frame in a video, recognize the face(s) and add to self.facesCount. Each frame is acquired by self.videoReader
            @param: numProcess: (int) 
            @param: yieldNum: (int)
            @param: sampleRate: (int)
            @param: k_certainty: (int) k should be larger than number of faces of any person. For 3 faces per person, the best practice is to set k=2
        """
        s = time.time()
        assert(numProcess > 0)
        

        # Check if all person has at least k_certainty faces, else raise exception
        cnt = {}
        for person in self.persons:
            name = person[:len(person) -1 - len(person.split("_")[-1])]
            
            if(name not in cnt):  # Add new key if not exist
                cnt[name] = 0
            cnt[name] += 1

        if(min(cnt.values()) < k_certainty):
            raise Exception("Minimum number of faces of each person is smaller than k_certainty which is", k_certainty, "!")



        frameRate = 29.97   # unit: frame/s
        videoLength = 45    # unit: min
        print("Total approximate number of frames:", len(self.videoReader.videoFiles) * videoLength * 60 * frameRate // sampleRate)
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
                # Data preperation (distribution)
                frames = np.array(frames)   # [20, shape of frame]

                # Find the maximum number of division in order to distribute tasks as much as possible
                for numTask in range(numProcess, 0, -1):
                    if(len(frames) % numTask == 0):
                        frames = frames.reshape(numTask, len(frames) // numTask, *frames.shape[1:])      # Remainder, e.g. [3, 3, shape of frame]
                        break
                
                # Create pool
                with Pool(numProcess) as pool:   
                    args = zip(frames, [self.facesEncodings]*numProcess, [self.persons]*numProcess, [k_certainty]*numProcess)    # Pack parameters  
                    resList = pool.starmap(self.batchedFacesRecogn, args)   # Distibute tasks to child processes

                for res in resList:     # Accumulate the returned results to master process
                    self.facesCount += res

            ee = time.time()
            print("Total time for processing", numFrames, "frames:", ee-ss, '\n')

        e = time.time()
        print("Total time:", e-s)

        # Calculate statistical result
        self._stat()
        print(self.getStat())


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

    def _stat(self):
        """
        Calculate statistical result from self.persons and self.facesCount
        """
        for nameCntPair in zip(self.persons, self.facesCount):
            name = nameCntPair[0]

            # 1.Split according to number
            # s = re.search(r'[0-9]', name)
            # if(bool(s)):    # Get rid of the number label for repeated faces
            #     name = name[0 : s.span()[0]]

            # 2.Split by "_"
            name = name[:len(name) -1 - len(name.split("_")[-1])]

            if(name not in self.stat):  # Add new key if not exist
                self.stat[name] = 0
            self.stat[name] += nameCntPair[1]

        
    def getStat(self):
        """
        Return statistical result as a dictionary
        """
        if(len(self.stat) == 0):
            print("No statistical result available. Invoke videoFaceRecogn() or directly invoke _stat() to get statistical result.")
        else:
            return self.stat
    

    def plotStat(self, figType='bar', saveFigName=None):
        """
        Plot statistical result as bar/pie chart
            @param: figType: ('bar' or 'pie')
            @param: saveFigName: (None or str)
        """
        sorted_stat = sorted(self.stat.items(), key=lambda pair: pair[1], reverse=True)
        name, count = zip(*sorted_stat)
        if(figType == 'bar'):
            plt.bar(name, count)
        elif(figType == 'pie'):
            plt.pie(name, count)

        if(saveFigName is not None):
            plt.savefig(saveFigName)