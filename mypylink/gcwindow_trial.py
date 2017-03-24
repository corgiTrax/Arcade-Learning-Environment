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

def updateCursor(cursor, loc, fgbm):
	'''Updates the content of the cursor'''
	
	if(fgbm != None):
		srcrct = cursor.get_rect().move(loc[0], loc[1])
		cursor.blit(fgbm, (0, 0), srcrct)
	

# TODO: ad-hoc class, to store the states required by drawgc()
class drawgc_wrapper:
	def __init__(self):
		self.pbackcursor = None
		self.cursorsize = (20, 20)
		self.cursor = Surface(self.cursorsize)
		self.cursor.fill((255, 255, 255, 255))
		draw.circle(self.cursor, (255, 0, 0), (self.cursorsize[0] // 2, self.cursorsize[1] // 2) , 12)
		self.backcursor = None
		self.srcrct = None
		self.prevrct = None
		self.oldv = None

		# TODO here I'm using a simple black box as the gaze cursor
		self.fgbm = Surface(self.cursorsize)
		self.fgbm.fill((255, 255, 255, 255))


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

		getEYELINK().flushKeybuttons(0)
		buttons = (0, 0)

	def drawgc(self, surf):
		'''Does gaze-contingent drawing; uses the getNewestSample() to get latest update '''

		dt = getEYELINK().getNewestSample() # check for new sample update
		if(dt != None):
			# Gets the gaze position of the latest sample,
			if self.eye_used == RIGHT_EYE and dt.isRightSample():
				gaze_position = dt.getRightEye().getGaze()
			elif self.eye_used == LEFT_EYE and dt.isLeftSample():
				gaze_position = dt.getLeftEye().getGaze()

			# Determines the top-left corner of the update region and determines whether an update is necessarily or not
			region_topleft = (gaze_position[0] - self.cursorsize[0] // 2, gaze_position[1] - self.cursorsize[1] // 2)
			if(self.oldv != None and self.oldv == region_topleft):
				# (code execution falls into here extremely often) gaze pos no change
				return
			self.oldv = region_topleft
			
			if(self.backcursor != None): #copy the current self.backcursor to the surface and get a new backup
				if(self.prevrct != None):	
					surf.blit(self.pbackcursor, (self.prevrct.x, self.prevrct.y))
					
				self.pbackcursor.blit(self.backcursor, (0, 0))
				self.pbackcursor.blit(self.backcursor, (0, 0))
				self.prevrct = self.srcrct.move(0, 0) #make a copy	
				self.srcrct.x = region_topleft[0]
				self.srcrct.y = region_topleft[1]
				self.backcursor.blit(surf, (0, 0), self.srcrct)
			
			else: # create a new self.backcursor and copy the new back cursor
				self.backcursor = Surface(self.cursorsize)
				self.pbackcursor = Surface(self.cursorsize)
				self.backcursor.fill((0, 255, 0, 255))
				self.srcrct = self.cursor.get_rect().move(0, 0)
				self.srcrct.x = region_topleft[0]
				self.srcrct.y = region_topleft[1]
				self.backcursor.blit(surf, (0, 0), self.srcrct)
				self.backcursor.blit(surf, (0, 0), self.srcrct)

			updateCursor(self.cursor, region_topleft, self.fgbm) # Updates the content of the cursor
			surf.blit(self.cursor, region_topleft) # Draws and shows the cursor content;
			display.flip()


def do_trial(surf, ale, play_beep_func):

	# This supplies the title at the bottom of the eyetracker display
	message = "record_status_message 'Trial %s'" % (ale.gamename)
	getEYELINK().sendCommand(message)	
	
	# Always send a TRIALID message before starting to record.
	# EyeLink Data Viewer defines the start of a trial by the TRIALID message.
	# This message is different than the start of recording message START that is logged when the trial recording begins. 
	# The Data viewer will not parse any messages, events, or samples, that exist in the data file prior to this message.
	msg = "TRIALID %s" % ale.gamename
	getEYELINK().sendMessage(msg)
	
	play_beep_func(0, repeat=5)
	getEYELINK().doDriftCorrect(surf.get_rect().w // 2, surf.get_rect().h // 2, 1, 1) 

	error = getEYELINK().startRecording(1, 1, 1, 1)
	if error:	return error
	gc.disable() # disable python garbage collection for the entire experiment
	try: 
		getEYELINK().waitForBlockStart(100,1,0) 
	except RuntimeError: 
		if getLastError()[0] == 0: # wait time expired without link data 
			end_trial()
			print ("ERROR: No link samples received!") 
			return TRIAL_ERROR 
		else: # for any other status simply re-raise the exception 
			raise
	surf.fill((255, 255, 255, 255))
	
	# according to pylink.chm:
	# "SYNCTIME" marks the zero-time in a trial. A number may follow, which 
	# is interpreted as the delay of the message from the actual stimulus onset. 
	# It is suggested that recording start 100 milliseconds before the display is
	# drawn or unblanked at zero-time, so that no data at the trial start is lost.
	getEYELINK().sendMessage("SYNCTIME %d" % 0) # From above doc it seems we can just send 0 because we haven't drawn anything yet
	dw = drawgc_wrapper()
	scr_recorder = ScreenRecorder(lambda:getEYELINK().trackerTime())
	ret_value = ale.run(dw.drawgc, scr_recorder.save, event_handler_callback_func) # experiment starts
	eyelink_err_code = post_experiment() # experiment ends
	if ret_value != 0: 
		eyelink_err_code = ret_value # ale.run()'s return value has higher priority than post_experiment()'s
	gc.enable()
	print "eyelink_err_code = %d" % eyelink_err_code
	return eyelink_err_code

def event_handler_callback_func(key_pressed):
	error = getEYELINK().isRecording() # First check if recording is aborted 
	if error != 0: # this will happen if the we click "abort trial" at host machine
		return True, error

	if key_pressed[K_ESCAPE]:
		print("Exitting the game...")
		getEYELINK().sendMessage("key_pressed non-atari esc")
	elif key_pressed[K_F1]:
		print("Pause the game...")
		getEYELINK().sendMessage("key_pressed non-atari pause")
	elif key_pressed[K_F5]:
		print("Calibrate....")
		getEYELINK().sendMessage("key_pressed non-atari calibrate")
	elif key_pressed[K_F7]:
		print("Showing gaze-contigent window....")
		getEYELINK().sendMessage("key_pressed non-atari gcwindow")

	return False, None

def post_experiment():
		end_trial()	
		# The TRIAL_RESULT message defines the end of a trial for the EyeLink Data Viewer. 
		# This is different than the end of recording message END that is logged when the trial recording ends. 
		# Data viewer will not parse any messages, events, or samples that exist in the data file after this message. 
		getEYELINK().sendMessage("TRIAL_RESULT %d" % 0)
		return getEYELINK().getRecordingStatus()
	
def end_trial():
	'''Ends recording: adds 100 msec of data to catch final events'''
	pumpDelay(100)
	getEYELINK().stopRecording()
	while getEYELINK().getkey():
		pass

def run_trials(rom_file, screen, play_beep_func):
	''' This function is used to run individual trials and handles the trial return values. '''

	''' Returns a successful trial with 0, aborting experiment with ABORT_EXPT (3); It also handles
	the case of re-running a trial. '''

	ale = aleForET(rom_file, screen)

	# Do the tracker setup at the beginning of the experiment.
	# Press ESC to skip eye tracker setup
	play_beep_func(2, repeat=2)
	getEYELINK().doTrackerSetup() 

	# Determine whether to redo the trail , finish the trial, or quit, depending on the return value
	# These return values are predefined in PyLink, so they might also be read by Data Viewer (I'm not sure).
	# Some return values checked below are never returned by do_trail(). 
	# But in the future do_trail() could return them when we need.
	while 1:
		ret_value = do_trial(screen, ale, play_beep_func)
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
		
