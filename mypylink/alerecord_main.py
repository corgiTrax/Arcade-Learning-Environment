from pylink import *
import time
import gc
import sys
import gcwindow_trial
from pygame import display
from EyeLinkCoreGraphicsPyGame import EyeLinkCoreGraphicsPyGame
from IPython import embed

try:
  eyelinktracker = EyeLink("100.1.1.1")
except Exception as ex:
	print(str(ex) + "\nEL is None, using dummy EyeLink interface")
        eyelinktracker = EyeLink(None)

if len(sys.argv) < 2:
	print 'Usage:', sys.argv[0], 'rom_file'
	sys.exit()
rom_file = sys.argv[1]

genv = EyeLinkCoreGraphicsPyGame(960,630,eyelinktracker)
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
	gcwindow_trial.set_play_beep_func(genv.play_beep2)
	gcwindow_trial.run_trials(rom_file, surf)

#Close the experiment graphics	
display.quit()
