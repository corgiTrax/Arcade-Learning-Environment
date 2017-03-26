#
# Copyright (c) 1996-2012, SR Research Ltd., All Rights Reserved
#
#
# For use by SR Research licencees only. Redistribution and use in source
# and binary forms, with or without modification, are NOT permitted.
#
#
#
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in
# the documentation and/or other materials provided with the distribution.
#
# Neither name of SR Research Ltd nor the name of contributors may be used
# to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS
# IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# $Date: 2012/10/25 19:04:00 $
# 
#

from pylink import *
from pygame import *
import time
import gc
import sys
from aleForET import aleForET
from IPython import embed
from ScreenRecorder import ScreenRecorder

#if you need to save bitmap features and/or backdrop features set
#BITMAP_SAVE_BACK_DROP to  true. This will require numpy or Numeric modules. Also
#in some configurations calling array3d segfaults. 
BITMAP_SAVE_BACK_DROP = False
if BITMAP_SAVE_BACK_DROP:
	from pygame.surfarray import *


RIGHT_EYE = 1
LEFT_EYE = 0
BINOCULAR = 2
DURATION = 20000

def set_play_beep_func(func): 
	global play_beep_func # used by below code to play a sound
	play_beep_func=func

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

def do_trail_subfunc_starting_msg(gamename):
	# This controls the title at the bottom of the eyetracker display
	getEYELINK().sendCommand("record_status_message 'Trial %s'" % (gamename))	
	# Always send a TRIALID message before starting to record.
	# EyeLink Data Viewer defines the start of a trial by the TRIALID message.
	# This message is different than the start of recording message START that is logged when the trial recording begins. 
	# The Data viewer will not parse any messages, events, or samples, that exist in the data file prior to this message.
	getEYELINK().sendMessage("TRIALID %s" % gamename)

def do_trail_subfunc_start_recording():
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

def do_trail_subfunc_starting_msg2():
	# according to pylink.chm:
	# "SYNCTIME" marks the zero-time in a trial. A number may follow, which 
	# is interpreted as the delay of the message from the actual stimulus onset. 
	# It is suggested that recording start 100 milliseconds before the display is
	# drawn or unblanked at zero-time, so that no data at the trial start is lost.
	getEYELINK().sendMessage("SYNCTIME %d" % 0) # From above doc it seems we can just send 0 because we haven't drawn anything yet

def do_trial(surf, ale):

	play_beep_func(0, repeat=5)
	try:
		getEYELINK().doDriftCorrect(surf.get_rect().w // 2, surf.get_rect().h // 2, 1, 0)
	except: # When ESC is pressed or "Abort" buttun clicked, an exception will be thrown
		pass

	do_trail_subfunc_starting_msg(ale.gamename)
	err = do_trail_subfunc_start_recording()
	if err != 0: return err
	do_trail_subfunc_starting_msg2()

	surf.fill((255, 255, 255, 255))
	getEYELINK().flushKeybuttons(0)
	gc.disable() # disable python garbage collection for the entire experiment

	# experiment starts
	dw = drawgc_wrapper()
	scr_recorder = ScreenRecorder(lambda:getEYELINK().trackerTime())
	eyelink_err_code = ale.run(dw.drawgc, scr_recorder.save, event_handler_callback_func)

	# experiment ends
	pumpDelay(100) # adds 100 msec of data to catch final events
	getEYELINK().stopRecording()
	while getEYELINK().getkey():
		pass
	# The TRIAL_RESULT message defines the end of a trial for the EyeLink Data Viewer. 
	# This is different than the end of recording message END that is logged when the trial recording ends. 
	# Data viewer will not parse any messages, events, or samples that exist in the data file after this message. 
	getEYELINK().sendMessage("TRIAL_RESULT %d" % 0)
	gc.enable()

	if eyelink_err_code == 0:
		eyelink_err_code = getEYELINK().getRecordingStatus() # Maybe this function has something to say (see document of this func)
	print "Trial Ended. eyelink_err_code = %d (%s)" % (eyelink_err_code, eyelink_err_code_to_str(eyelink_err_code))
	return eyelink_err_code

bool_drawgc = True
def event_handler_callback_func(key_pressed):
	global bool_drawgc
	# First check if host PC is still recording
	# This will block the thread when "abort trial" is clicked at host PC and the "abort trail" menu is shown 
	error = getEYELINK().isRecording() 
	if error != 0: # happens when "abort trial" is clicked at host PC
		return True, error, bool_drawgc

	if key_pressed[K_ESCAPE]:
		print("Exiting the game...")
		getEYELINK().sendMessage("key_pressed non-atari esc")
		return True, SKIP_TRIAL, bool_drawgc
	elif key_pressed[K_F1]:
		print("Pause the game...")
		getEYELINK().sendMessage("key_pressed non-atari pause")
		getEYELINK().doTrackerSetup()
	elif key_pressed[K_F5]:
		print("Calibrate....")
		getEYELINK().sendMessage("key_pressed non-atari calibrate")
		getEYELINK().doTrackerSetup()
	elif key_pressed[K_F7]:
		print("Showing gaze-contigent window....")
		getEYELINK().sendMessage("key_pressed non-atari gcwindowON")
		bool_drawgc = True
	elif key_pressed[K_F8]:
		print("Hiding gaze-contigent window....")
		getEYELINK().sendMessage("key_pressed non-atari gcwindowOFF")
		bool_drawgc = False

	return False, None, bool_drawgc

def run_trials(rom_file, screen):
	''' This function is used to run all trials and handles the return value of each trial. '''

	ale = aleForET(rom_file, screen)

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
			return ABORT_EXPT
		elif (ret_value == REPEAT_TRIAL):
			getEYELINK().sendMessage("TRIAL REPEATED")
		else: 
			getEYELINK().sendMessage("TRIAL ERROR")
			break

	return 0
		
def eyelink_err_code_to_str(code):
	if (code == TRIAL_OK):
		return "TRIAL OK"
	elif (code == SKIP_TRIAL):
		return "TRIAL ABORTED"
	elif (code == ABORT_EXPT):
		return "EXPERIMENT ABORTED"
	elif (code == REPEAT_TRIAL):
		return "TRIAL REPEATED"
	else:
		return "TRIAL ERROR"

