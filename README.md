# Movie-Role-Analyzer
This project analyzes the importance of character role in a movie/series/video by applying face recognition frame-wise and output the statistical result of appearance of each role.



## Version 1.2.0 

![](./design/ClassDiagram_v1.0.0.png)

### Features:

1. `VideoReader` for reading frames of each movie in a given directory.

2. `FaceRecognition` for analyzing each frame with the ground truth faces

   

### Problems: 

1. Efficiency: only can process about 1~3 frames per second
2. Accuracy: only frontal faces are recognized
3. What to do if nobody's face is matched?
4. :white_check_mark:Functionality: statistical result visualization at the end
5. Functionality: skip frames at the beginning and end



### Solutions:

* Efficiency

	1. :white_check_mark:Frame sampling with a sample rate of 3~10 
	2. :white_check_mark:Resolution down-sampling 
	3. :white_check_mark:Parallelize code (e.g. pipelined reader-analyzer, data parallelism) -> speedup = 2 for 4 processes
	4. C++ implementation (version 3.0)
* Accuracy
  1. Unmatched faces (Long Tao)
  2. :white_check_mark:More (repeated) ground truth faces to decrease uncertainty
  3. CNN (version 2.0)
  4. Temporal smoothing

