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
	
	
def end_trial():
	'''Ends recording: adds 100 msec of data to catch final events'''
	pylink.endRealTimeMode()
	pumpDelay(100)
	getEYELINK().stopRecording()
	while getEYELINK().getkey():
		pass

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
		# TODO return statement are all not drawgc()'s responsibility. move to somewhere else.

		
		error = getEYELINK().isRecording()# First check if recording is aborted 
		if error != 0:
			# TODO if the we terminate recording at the Host machine, code execution 
			# will always fall into here.
			end_trial()
			print "1", error
			return error

		# TODO maybe add back "duration" feature later
		# if (currentTime() - startTime) > DURATION:#Writres out a time out message if no response is made
		# 	getEYELINK().sendMessage("TIMEOUT")
		# 	end_trial()
		# 	buttons = (0, 0)
		# 	break
		
		if(getEYELINK().breakPressed()):	# Checks for program termination or ALT-F4 or CTRL-C keys
			end_trial()
			print "2"
			return ABORT_EXPT
		#elif(getEYELINK().escapePressed()): # Checks for local ESC key to abort trial (useful in debugging)
		#	end_trial()
		#	print "3"
		#	return SKIP_TRIAL
			
		buttons = getEYELINK().getLastButtonPress() # Checks for eye-tracker buttons pressed
		if(buttons[0] != 0):
			getEYELINK().sendMessage("ENDBUTTON %d" % (buttons[0]))
			end_trial()
			print "4"
			return ABORT_EXPT		
			
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
				return  # I DON'T know what to return here. there is no error here.
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

			updateCursor(self.cursor, region_topleft, self.fgbm)#Updates the content of the cursor
			surf.blit(self.cursor, region_topleft)#Draws and shows the cursor content;
			display.flip()

		# (code execution falls into here extremely often)
		return getEYELINK().getRecordingStatus()


	# TODO move to somewhere else. Not a responsibility of drawgc()
	def post_experiment(self):
		end_trial()	
		#The TRIAL_RESULT message defines the end of a trial for the EyeLink Data Viewer. 
		#This is different than the end of recording message END that is logged when the trial recording ends. 
		#Data viewer will not parse any messages, events, or samples that exist in the data file after this message. 
		getEYELINK().sendMessage("TRIAL_RESULT %d" % (buttons[0]))
		return getEYELINK().getRecordingStatus()




fgtext = [
"Buck did not read the newspapers, or he would have known that",
"trouble was brewing, not alone for himself, but for every ",
"tide-water dog, strong of muscle and with warm, long hair, from",
"Puget Sound to San Diego. Because men, groping in the Arctic ",
"darkness, had found a yellow metal and because steamship and ",
"transportation companies were booming the find, thousands of ",
"men were rushing into the Northland. These men wanted dogs, ",
"and the dogs they wanted were heavy dogs, with strong muscles ",
"by which to toil, and furry coats to protect them from the frost.",
"																 ",
"Buck lived at a big house in the sun-kissed Santa Clara ",
"Valley. Judge Miller's place, it was called. It stood back ",
"from the road, half hidden among the trees, through which ",
"glimpses could be caught of the wide cool veranda that ran ",
"around its four sides."
]

bgtext = [
"Xxxx xxx xxx xxxx xxx xxxxxxxxxxx xx xx xxxxx xxxx xxxxx xxxx",
"xxxxxxx xxx xxxxxxxx xxx xxxxx xxx xxxxxxxx xxx xxx xxxxx ",
"xxxxxxxxxx xxxx xxxxxx xx xxxxxx xxx xxxx xxxxx xxxx xxxxx xxxx",
"Xxxxx Xxxxx xx Xxx Xxxxxx Xxxxxxx xxxx xxxxxxx xx xxx Xxxxxx ",
"xxxxxxxxx xxx xxxxx x xxxxxx xxxxx xxx xxxxxxx xxxxxxxxx xxx ",
"xxxxxxxxxxxxxx xxxxxxxxx xxxx xxxxxxx xxx xxxxx xxxxxxxxx xx ",
"xxx xxxx xxxxxxx xxxx xxx Xxxxxxxxxx Xxxxx xxx xxxxxx xxxxx ",
"xxx xxx xxxx xxxx xxxxxx xxxx xxxxx xxxxx xxxx xxxxxx xxxxxxx ",
"xx xxxxx xx xxxx, xxx xxxxx xxxxx xx xxxxxxx xxxx xxxx xxx xxxxxx",
"																 ",
"Xxxx xxxxx xx x xxx xxxxx xx xxx xxxxxxxxxx Xxxxx Xxxxx ",
"Xxxxxxx Xxxxx Xxxxxxxx xxxxxx xx xxx xxxxxxx Xx xxxxx xxxx ",
"xxxx xxx xxxxx xxxx xxxxxx xxxxx xxx xxxxxx xxxxxxx xxxxx ",
"xxxxxxxx xxxxx xx xxxxxx xx xxx xxxx xxxx xxxxxxx xxxx xxx ",
"xxxxxx xxx xxxx xxxxxx"
]



def getTxtBitmap(text, dim):
	''' This function is used to create a page of text. '''

	''' return image object if successful; otherwise None '''

	if(not font.get_init()):
		font.init()
	fnt = font.Font("cour.ttf", 15)
	fnt.set_bold(1)
	sz = fnt.size(text[0])
	bmp = Surface(dim)
	
	bmp.fill((255, 255, 255, 255))
	for i in range(len(text)):
		txt = fnt.render(text[i], 1, (0, 0, 0, 255), (255, 255, 255, 255))
		bmp.blit(txt, (0, sz[1] * i))
	
	return bmp
	
	
def getImageBitmap(pic):
	''' This function is used to load an image into a new surface. '''

	''' return image object if successful; otherwise None '''

	if(pic == 1):
		try:
			bmp = image.load("sacrmeto.jpg", "jpg")
			return bmp
		except:
			print("Cannot load image sacrmeto.jpg")
			return None
	else:
		try:
			bmp = image.load("sac_blur.jpg", "jpg")
			return bmp
		except:
			print("Cannot load image sac_blur.jpg")
			return None
	
	
	
def arrayToList(w, h, dt):
	rv = []
	for y in range(h):
		line = []
		for x in range(w):
			v = dt[x, y]
			line.append((v[0], v[1], v[2]))
		rv.append(line)
	return rv
def do_trial(surf, ale):
	'''Does the simple trial'''

	#This supplies the title at the bottom of the eyetracker display
	message = "record_status_message 'Trial %s'" % (ale.gamename)
	getEYELINK().sendCommand(message)	
	
	#Always send a TRIALID message before starting to record.
	#EyeLink Data Viewer defines the start of a trial by the TRIALID message.
	#This message is different than the start of recording message START that is logged when the trial recording begins. 
	#The Data viewer will not parse any messages, events, or samples, that exist in the data file prior to this message.
	msg = "TRIALID %s" % ale.gamename
	getEYELINK().sendMessage(msg)
	
		
	# Now we don't have such image. Just skip these code
	# The following code is for the EyeLink Data Viewer integration purpose. 
	# See section "Protocol for EyeLink Data to Viewer Integration" of the EyeLink Data Viewer User Manual
	# The IMGLOAD command is used to show an overlay image in Data Viewer 
	## getEYELINK().sendMessage("!V IMGLOAD FILL  sacrmeto.jpg") 
	
	# This TRIAL_VAR command specifies a trial variable and value for the given trial. 
	# Send one message for each pair of trial condition variable and its corresponding value.
	## getEYELINK().sendMessage("!V TRIAL_VAR image  sacrmeto.jpg")
	## getEYELINK().sendMessage("!V TRIAL_VAR type  gaze_contingent")
	 
	
	# if BITMAP_SAVE_BACK_DROP:
	# 	#array3d(bgbm) crashes on some configurations. 
	# 	agc = arrayToList(bgbm.get_width(), bgbm.get_height(), array3d(bgbm))
	# 	bitmapSave(bgbm.get_width(), bgbm.get_height(), agc, 0, 0, bgbm.get_width(), bgbm.get_height(), "trial" + str(trial) + ".bmp", "trialimages", SV_NOREPLACE,)
	# 	getEYELINK().bitmapSaveAndBackdrop(bgbm.get_width(), bgbm.get_height(), agc, 0, 0, bgbm.get_width(), bgbm.get_height(), "trial" + str(trial) + ".png", "trialimages", SV_NOREPLACE, 0, 0, BX_MAXCONTRAST)
	
	
	# TODO: ??? WHEN DOES A DRIFT CORRECTION RETURNS 0 ? BY PRESSING ENTER KEY OR SAPCE KEY ???
	#The following does drift correction at the begin of each trial
	while True: 
		# Checks whether we are still connected to the tracker
		if not getEYELINK().isConnected():
			return ABORT_EXPT			
		
		# Does drift correction and handles the re-do camera setup situations
		try:
			error = getEYELINK().doDriftCorrect(surf.get_rect().w // 2, surf.get_rect().h // 2, 1, 1)
			if error != 27: 
				break
			else:
				getEYELINK().doTrackerSetup()
		except:
			getEYELINK().doTrackerSetup()		
	
	#switch tracker to ide and give it time to complete mode switch
	getEYELINK().setOfflineMode()
	msecDelay(50) 


	error = getEYELINK().startRecording(1, 1, 1, 1)
	if error:	return error
	gc.disable()
	#begin the realtime mode
	#TODO: check beginRealTimeMode, the doc says it's relevant to realtime running mode in windows?
	#pylink.beginRealTimeMode(100)
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
	startTime = currentTime()
	
	getEYELINK().sendMessage("SYNCTIME %d" % (currentTime() - startTime))
	dw = drawgc_wrapper()
	ale.run(dw.drawgc)
	ret_value = drawgc_wrapper.post_experiment()
	# pylink.endRealTimeMode()
	gc.enable()
	return ret_value

def drawgc_dummy(surf):
	pass
	
def run_trials(rom_file, screen):
	''' This function is used to run individual trials and handles the trial return values. '''

	''' Returns a successful trial with 0, aborting experiment with ABORT_EXPT (3); It also handles
	the case of re-running a trial. '''

	ale = aleForET(rom_file, screen)
	#Do the tracker setup at the beginning of the experiment.
	# Press ESC to skip eye tracker setup
	getEYELINK().doTrackerSetup() 

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
		
