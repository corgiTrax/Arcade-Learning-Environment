
# The ATARI games' screen size is (w,h) = (160*210)
# But when humans play, we scale it according to the following factors.
# *Importance* Therefore, gaze position data is recorded on a screen of size (w*xSCALE, h*ySCALE). And since the neural network is trained on unscaled images (i.e. of size (w,h)), the data preprocess code will use the following constants to map recorded gaze positon back to an unscaled version, that is, by computing (x/xSCALE, y/ySCALE) and feed the result into neural network training. 
NUM_ACTION = 18
xSCALE, ySCALE = 8, 4 # was 6,3
SCR_W, SCR_H = 160*xSCALE, 210*ySCALE
