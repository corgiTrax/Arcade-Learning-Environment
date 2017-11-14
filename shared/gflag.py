"""
GFlag stands for "global flag".

Example usage: 
  from gflag import gflag
  gflag.dueling 

This file has two use cases:
+ Storing command line flags(i.e arguments)
  so that they can be universally accessible after importing this file.
+ Storing additional "Global Constants" so that they can be universally accessible.
  WARNING: don't abuse this use case. You should store things other than flags
  very sparingly, because it is uncertain to the code reader that when is this 
  variable available. and thus deteriorates readability. It is mostly useful
  for DEBUGGING, where you need to make a temparory var inside a function
  become readable somewhere else (e.g. your debug function) .
"""
import sys

class GFlag(object): 
  _dict = None # None denotes the uninitialized state

  def init_me_as(self, argsdict):
    if GFlag._dict is None:
      GFlag._dict = argsdict
    else:
      raise AttributeError("GFlag is already initialized")

  def add_read_only(self, name, val):
    if GFlag._dict is None:
      GFlag._dict = {} # initialize
    if name not in GFlag._dict:
      GFlag._dict[name] = val
    elif GFlag._dict[name] != val:
      raise AttributeError("GFlag is immutable after initialization")

  def exists(self, name):
    return (GFlag._dict is not None) and (name in GFlag._dict)

  def __getattr__(self, name):
    if GFlag._dict is None:
      raise AttributeError("Flag named '%s' not found: GFlag hasn't been initialized." % name)
    if name not in GFlag._dict:
      raise AttributeError("Flag named '%s' not found in GFlag" % name)
    return GFlag._dict[name]

  def __setattr__(self, name, val):
    raise AttributeError("GFlag is immutable after initialization")
    

gflag = GFlag() # Allow `from gflag import gflag`. Works in both Python 2 and 3
