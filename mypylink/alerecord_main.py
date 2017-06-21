import subprocess, random, gc, sys, time as T # avoid collision with pygame.time
from EyeLinkCoreGraphicsPyGame import EyeLinkCoreGraphicsPyGame
from pylink import *
from pygame import *
from aleForET import aleForET
from IPython import embed
from ScreenRecorder import ScreenRecorder
import vip_constants as V
import action_enums


RIGHT_EYE = 1
LEFT_EYE = 0
BINOCULAR = 2
step_by_step_mode = None
unique_trial_id = None
scr_recorder = None


# A class used to store the states required by drawgc()
class drawgc_wrapper:
	def __init__(self):
		self.cursorsize = (15, 15)
		self.cursor = Surface(self.cursorsize)
		self.cursor.fill((255, 255, 255, 255))
		draw.circle(self.cursor, (255, 0, 0), (self.cursorsize[0] // 2, self.cursorsize[1] // 2) , 5)

		eye_used = getEYELINK().eyeAvailable() #determine which eye(s) are available 
		# TODO do we only track one eye?
		if eye_used == RIGHT_EYE:
			getEYELINK().sendMessage("EYE_USED 1 RIGHT")
		elif eye_used == LEFT_EYE or eye_used == BINOCULAR:
			getEYELINK().sendMessage("EYE_USED 0 LEFT")
			eye_used = LEFT_EYE
		else:
			raise Exception("Error in getting the eye information!")
		self.eye_used = eye_used

	def drawgc(self, surf):
		'''Does gaze-contingent drawing; uses getNewestSample() to get latest update '''
		dt = getEYELINK().getNewestSample() # check for new sample update
		if(dt != None):
			# Gets the gaze position of the latest sample
			if self.eye_used == RIGHT_EYE and dt.isRightSample():
				gaze_position = dt.getRightEye().getGaze()
			elif self.eye_used == LEFT_EYE and dt.isLeftSample():
				gaze_position = dt.getLeftEye().getGaze()

			# Determines the top-left corner
			region_topleft = (gaze_position[0] - self.cursorsize[0] // 2, gaze_position[1] - self.cursorsize[1] // 2)

			surf.blit(self.cursor, region_topleft) # Draws and shows the cursor content;

def do_trial_subfunc_starting_msg(gamename):
	# This controls the title at the bottom of the eyetracker display
	getEYELINK().sendCommand("record_status_message 'Trial %s'" % (gamename))	
	# Always send a TRIALID message before starting to record.
	# EyeLink Data Viewer defines the start of a trial by the TRIALID message.
	# This message is different than the start of recording message START that is logged when the trial recording begins. 
	# The Data viewer will not parse any messages, events, or samples, that exist in the data file prior to this message.
	getEYELINK().sendMessage("TRIALID %s" % gamename)

def do_trial_subfunc_start_recording():
	error = getEYELINK().startRecording(1, 1, 1, 1)
	if error:	return error
	try: 
		getEYELINK().waitForBlockStart(100,1,0) 
	except RuntimeError: 
		if getLastError()[0] == 0: # wait time expired without link data 
			getEYELINK().stopRecording()
			print ("ERROR: No link samples received!") 
			return TRIAL_ERROR 
		else: # for any other status simply re-raise the exception 
			raise
	return 0

def do_trial_subfunc_starting_msg2():
	# according to pylink.chm:
	# "SYNCTIME" marks the zero-time in a trial. A number may follow, which 
	# is interpreted as the delay of the message from the actual stimulus onset. 
	# It is suggested that recording start 100 milliseconds before the display is
	# drawn or unblanked at zero-time, so that no data at the trial start is lost.
	getEYELINK().sendMessage("SYNCTIME %d" % 0) # From above doc it seems we can just send 0 because we haven't drawn anything yet

def do_trial_subfunc_end_recording():
	pumpDelay(100) # adds 100 msec of data to catch final events
	getEYELINK().stopRecording()
	while getEYELINK().getkey():
		pass
	# The TRIAL_RESULT message defines the end of a trial for the EyeLink Data Viewer. 
	# This is different than the end of recording message END that is logged when the trial recording ends. 
	# Data viewer will not parse any messages, events, or samples that exist in the data file after this message. 
	getEYELINK().sendMessage("TRIAL_RESULT %d" % 0)

def do_trial(surf, ale):

	play_beep_func(0, repeat=5)
	try:
		print "Now performing: drift correction"
		getEYELINK().doDriftCorrect(surf.get_rect().w // 2, surf.get_rect().h // 2, 1, 0)
	except: # When ESC is pressed or "Abort" buttun clicked, an exception will be thrown
		pass

	do_trial_subfunc_starting_msg(ale.gamename)
	err = do_trial_subfunc_start_recording()
	if err != 0: return err
	do_trial_subfunc_starting_msg2()

	surf.fill((255, 255, 255, 255))
	getEYELINK().flushKeybuttons(0)
	event.pump() # discard all events (usually key events) prior to experiment
	gc.disable() # disable python garbage collection for the entire experiment

	# experiment starts
	global scr_recorder
	scr_recorder=ScreenRecorder(unique_trial_id)
	dw = drawgc_wrapper()
	if step_by_step_mode:
		eyelink_err_code = ale.run_in_step_by_step_mode(dw.drawgc, save_screen_callback_func, event_handler_callback_func, record_a_and_r_callback_func)
	else:
		eyelink_err_code = ale.run(dw.drawgc, save_screen_callback_func, event_handler_callback_func, record_a_and_r_callback_func)

	# experiment ends
	gc.enable()
	do_trial_subfunc_end_recording()
	if eyelink_err_code == 0:
		eyelink_err_code = getEYELINK().getRecordingStatus() # This function has something to say, so get its err_code (see api doc for details)  

	# After the recording is done, we should run "Validation".
	# So here doTrackerSetup() is called, and then the experimenter clicks "Validate". If the validation result is 
	# good, do nothing. If bad, the experimenter marks this experiment as "bad" on paper. (the code won't do the marking, for now)
	play_beep_func(1, repeat=5)
	print "Switched the eye tracker to Setup menu. Now you may perform validation."
	getEYELINK().doTrackerSetup()

	print "Trial Ended. eyelink_err_code = %d (%s) Compressing recorded frames..." % (eyelink_err_code, eyelink_err_code_to_str(eyelink_err_code))
	scr_recorder.compress_record_dir(remove_original_dir=True)
	return eyelink_err_code


def save_screen_callback_func(screen, frameid):
	# the drawing of current frame should align with the recording of eye
	# position as close as possible. So we put sendMessage at the first line.
	getEYELINK().sendMessage("SCR_RECORDER FRAMEID %d UTID %s" % (frameid, unique_trial_id))
	global scr_recorder
	scr_recorder.save(screen, frameid)

bool_drawgc = False
def event_handler_callback_func(key_pressed, caller):
	global bool_drawgc
	# First check if host PC is still recording
	# This will block the thread when "abort trial" is clicked at host PC and the "abort trial" menu is shown 
	error = getEYELINK().isRecording() 
	if error != 0: # happens when "abort trial" is clicked at host PC
		return True, error, bool_drawgc

	if key_pressed[K_ESCAPE]:
		print("Exiting the game...")
		getEYELINK().sendMessage("key_pressed non-atari esc")
		return True, SKIP_TRIAL, bool_drawgc
	elif key_pressed[K_F1]:
		fname = "saved_games/%s.npy" % (unique_trial_id)
		print "Saving the game to %s ..." % fname
		getEYELINK().sendMessage("key_pressed non-atari save")
		alestate = caller.saveALEState(fname)
	elif key_pressed[K_F7]:
		print("Showing gaze-contigent window....")
		getEYELINK().sendMessage("key_pressed non-atari gcwindowON")
		bool_drawgc = True
	elif key_pressed[K_F8]:
		print("Hiding gaze-contigent window....")
		getEYELINK().sendMessage("key_pressed non-atari gcwindowOFF")
		bool_drawgc = False

	return False, None, bool_drawgc

def record_a_and_r_callback_func(atari_action, reward):
	getEYELINK().sendMessage("key_pressed atari_action %d" % (atari_action))
	getEYELINK().sendMessage("reward %d" % (reward))

def run_trials(rom_file, screen, resume_state_file):
	''' This function is used to run all trials and handles the return value of each trial. '''

	rndseed = random.randint(0,65535)
	ale = aleForET(rom_file, screen, rndseed, resume_state_file)
	getEYELINK().sendMessage("random_seed %d" % (rndseed))

	# Show tracker setup screen at the beginning of the experiment.
	# It won't return unitl we press ESC on display PC or click "Exit Setup" on host PC
	play_beep_func(2, repeat=2)
	getEYELINK().doTrackerSetup()

	# Determine whether to redo/finish/quit the trial, depending on the return value
	# These return values are predefined in PyLink, so they might also be read by Data Viewer (I'm not sure).
	# Some are returned when (for example) we click "Abort Trial" and then "Abort Experiment" or "Repeat Trial" on host PC
	while 1:
		ret_value = do_trial(screen, ale)
		endRealTimeMode()
	
		if (ret_value == TRIAL_OK):
			getEYELINK().sendMessage("TRIAL OK")
			break
		elif (ret_value == SKIP_TRIAL):
			getEYELINK().sendMessage("TRIAL ABORTED")
			break			
		elif (ret_value == ABORT_EXPT):
			getEYELINK().sendMessage("EXPERIMENT ABORTED")
			break
		elif (ret_value == REPEAT_TRIAL):
			getEYELINK().sendMessage("TRIAL REPEATED")
		else: 
			getEYELINK().sendMessage("TRIAL ERROR")
			break

	# Experiment ended
	if getEYELINK() != None:
		# File transfer and cleanup!
		getEYELINK().setOfflineMode();                          
		msecDelay(500);                 

		#Close the file and transfer it to Display PC
		getEYELINK().closeDataFile()
		getEYELINK().receiveDataFile(edfFileName, scr_recorder.dir+".edf")
		getEYELINK().close();

		print "Calling command edf2asc..."
		if subprocess.call('edf2asc %s' % (scr_recorder.dir+".edf"), shell=True) != 0:
			print "\nERROR: Non-zero exit status was returned. See errors above."

	return 0
		
def eyelink_err_code_to_str(code):
	if (code == TRIAL_OK):
		return "TRIAL OK"
	elif (code == SKIP_TRIAL):
		return "TRIAL ABORTED (SKIP_TRIAL)"
	elif (code == ABORT_EXPT):
		return "EXPERIMENT ABORTED (ABORT_EXPT)"
	elif (code == REPEAT_TRIAL):
		return "TRIAL REPEATED (REPEAT_TRIAL)"
	else:
		return "TRIAL ERROR"

def set_play_beep_func(func): 
	global play_beep_func # used by below code to play a sound
	play_beep_func=func

if __name__ == "__main__":
	try:
		eyelinktracker = EyeLink("100.1.1.1")
	except Exception as ex:
		print(str(ex) + '\nEyeLink("100.1.1.1") is None, using dummy EyeLink interface')
		eyelinktracker = EyeLink(None)

	if len(sys.argv) < 4:
		print 'Usage:', sys.argv[0], 'rom_file', "step_by_step_mode(true|false)", "expr_name", "[resume_state_file]"
		sys.exit()
	rom_file = sys.argv[1]
	step_by_step_mode = sys.argv[2].upper() == "TRUE"
	unique_trial_id = "%s_%d" % (sys.argv[3], int(T.time())%10000000)
	resume_state_file = sys.argv[4] if len(sys.argv) > 4 else None

	genv = EyeLinkCoreGraphicsPyGame(160*V.xSCALE,210*V.ySCALE,eyelinktracker)
	openGraphicsEx(genv)

	#Opens the EDF file.
	edfFileName = "ATARI.EDF";
	getEYELINK().openDataFile(edfFileName)		
		
	flushGetkeyQueue(); 
	getEYELINK().setOfflineMode();                          

	#Gets the display surface and sends a mesage to EDF file;
	surf = display.get_surface()
	getEYELINK().sendCommand("screen_pixel_coords =  0 0 %d %d" %(surf.get_rect().w - 1, surf.get_rect().h - 1))
	getEYELINK().sendMessage("DISPLAY_COORDS  0 0 %d %d" %(surf.get_rect().w - 1, surf.get_rect().h - 1))

	tracker_software_ver = 0
	eyelink_ver = getEYELINK().getTrackerVersion()
	if eyelink_ver == 3:
		tvstr = getEYELINK().getTrackerVersionString()
		vindex = tvstr.find("EYELINK CL")
		tracker_software_ver = int(float(tvstr[(vindex + len("EYELINK CL")):].strip()))
		

	if eyelink_ver>=2:
		getEYELINK().sendCommand("select_parser_configuration 0")
		if eyelink_ver == 2: #turn off scenelink camera stuff
			getEYELINK().sendCommand("scene_camera_gazemap = NO")
	else:
		getEYELINK().sendCommand("saccade_velocity_threshold = 35")
		getEYELINK().sendCommand("saccade_acceleration_threshold = 9500")
		
	# set EDF file contents 
	getEYELINK().sendCommand("file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT")
	if tracker_software_ver>=4:
		getEYELINK().sendCommand("file_sample_data  = LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS,HTARGET,INPUT")
	else:
		getEYELINK().sendCommand("file_sample_data  = LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS,INPUT")

	# set link data (used for gaze cursor) 
	getEYELINK().sendCommand("link_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,INPUT")
	if tracker_software_ver>=4:
		getEYELINK().sendCommand("link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,HTARGET,INPUT")
	else:
		getEYELINK().sendCommand("link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT")
		
		
	# Start the experiment!

	if(getEYELINK().isConnected() and not getEYELINK().breakPressed()):
		set_play_beep_func(genv.play_beep2)
		run_trials(rom_file, surf, resume_state_file)

	#Close the experiment graphics	
	display.quit()
