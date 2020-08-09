# Movie-Role-Analyzer
This project analyzes the importance of character role in a movie/series/video by applying face recognition frame-wise and output the statistical result of appearance of each role.



## Version 1.0.0 

![](./design\ClassDiagram_v1.0.0.png)

### Features:

1. `VideoReader` for reading frames of each movie in a given directory.

2. `FaceRecognition` for analyzing each frame with the ground truth faces

   

### Problems: 

1. Efficiency: only can process about 1~3 frames per second
2. Accuracy: only frontal faces are recognized
3. What to do if nobody's face is matched?



### Solutions:

* Efficiency

	1. Frame sampling with a sample rate of 3~10​ :white_check_mark:
	2. Resolution down-sampling :white_check_mark:
	3. Parallelize code (e.g. pipelined reader-analyzer)
	4. C++ implementation
* Accuracy
	1. More (repeated) ground truth faces to decrease uncertainty
	2. CNN

