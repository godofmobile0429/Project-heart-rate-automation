Hello, sir

Please read this document first for application

Preparation:
1. You should additionally install packages: They are
	imutils, pandas, cydets
2. Please run this program with the follwing command in CMD or PowerShell

	"python opencv_object_tracking.py --video test.mp4 --tracker csrt"
	
	in this command
		test.mp4: the file to be opened
		csrt: tracking method
3. You can see the running process in your screen
	If you want to select ROI region, you should press 's' and to resume, press 'space' or 'enter'.
	This shows the heart rate graph in real-time.
	And after surfing the whole video, it will show the graph of the heart rate 
	And save the result in the 'result_bps.csv' file

Project Implementation method:
	I have calculated Euclidean distance between frames from that series, I have gained the cycle using cydets algorithm.
	It is the best algorithm to optimize the cycle of time-series
	After getting cycle from data, I have calculated the heart rate according to each cycles


If you have got a problem, please contact in any time.


Thanks!