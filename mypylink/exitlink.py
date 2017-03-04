#!/usr/bin/env python
import os
import time
from pylink import *
os.system('pkill -9 ipython')
time.sleep(1)

e=EyeLink('100.1.1.1')
getEYELINK().setOfflineMode();
msecDelay(500);

getEYELINK().closeDataFile()
getEYELINK().close();

