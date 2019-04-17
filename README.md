# To do
1. Dataset 
- [x] Record actions in to EDF file 
- [x] Save and reload game
- [x] Game time limits (15 minutes per session then break)
- [x] Validation after each trial, mark and throw away bad data
- [x] Score leaderboard
- [x] Automatically run edf2asc after each trial
- [x] Record random seed to data file
- [ ] A recording schedule for diff. games and subjects
- [ ] Support for composed actions using event detection

2. Imitation
- [x] Make it easier to combine trials(see dataset\_specification\_example.txt)
- [ ] Test regularizer hypothesis (attention as) 
- [x] Gaze-centered images as training samples
- [x] Make sure dropout is turned off during evaluation
- [x] Figure out Tau (needs to find gaze & image before easier)
- [x] CNN + past X frames model
- [ ] CNN + positional encoding
- [ ] RNN model
- [x] Foveated rendering model
- [ ] Cortical expansion model
- [ ] Log-polar transfomation
- [ ] Python implementation of the above ones if necessary

3. Gaze modeling
- [ ] CNN - optical flow; CNN - IttiKoch
- [ ] CNN - dilation model
- [x] CNN - deconv model
- [x] CNN - regression model
- [ ] Superior colliculus model
- [ ] KL epsilon regularizatin

4. AI playing 
- [x] Make it possible for model to play the game and record scores

  Software Archetecture
  + Three separate components:  Model definition, Model training, ALE game control
  + Instead of letting one of these components manage the others, use a "main" file to assemble and manage them. Because we are probably going to use complex ways to train a model in the future: SL training, RL training, SL pretraining + RL training, interleaved SL + RL training, switching off training (pure game playing), etc. These 'strategies' can be put into the "main" file while keeping these three components intact.
  
  To run it, you need to have a machine with tensorflow, keras, ale_python_interface installed, and **a monitor** (so eldar-11 cannot run it). And then run:

  ```
    # run this command to see help messages
    python runai.py 

    # Run baseline model playing seaquest trained on {54 62 67 83 86 87}tr_{70 75}val
    ipython runai.py -- ../roms/seaquest.bin BaselineModel baseline_actionModels/seaquest.hdf5 Img+OF_gazeModels/seaquest.mean.npy

    # Run a 'PastKFrameOpticalFlowGaze_and_CurrentFrameAction' model playing seaquest trained on {54 62 67 83 86 87}tr_{70 75}val
    ipython runai.py -- ../roms/seaquest.bin PastKFrameOpticalFlowGaze_and_CurrentFrameAction PreMul-2ch_actionModels/seaquest.gazeIsFrom.Img+OF.hdf5 Img+OF_gazeModels/seaquest.mean.npy == 4 1 0 Img+OF_gazeModels/seaquest.hdf5 Img+OF_gazeModels/seaquest.of.mean.npy

    # Other available models are "PastKFrameModel", "PastKFrameGaze_and_CurrentFrameAction".
    # But their models fils are not stored in repo due to repo size limit. 

    # Change the above runai.py to runai_noScrSupport.py to run the game without GUI. It's faster and used for evaluating AI.
  ```
  
  Available keyboard controls: 
  
  + h: Human(you) takes over the control of the game
  + p: print action's logit output 
  + x: Run `embed()` for debugging 
  + Esc: quit 

5. Psychology
- [x] Ask experts to validate experimental setups
- [ ] Experiment and config class
- [ ] Demographical information survey (ask Sariel)
- [ ] Subject consensus files (ask Sariel) 
- [ ] Organize experimental procedure
- [ ] Practice game for subjects + instructions
- [ ] Write experimental instruction for both experimentor and subjects; note that experimentor should center the screen; experimentor should stay with subjects during experiment 
- [ ] automatically run edf2asc after each trial
- [ ] Record random seed to data file
- [x] Practice game for subjects (without calling record and eye tracking)
- [ ] Practice game instructions
- [ ] ALE + Keras game playing agent 
- [x] ALE save/load
- [ ] What's the framerate's effect on gaze distribution ???

## Next

[![Build Status](https://travis-ci.org/mgbellemare/Arcade-Learning-Environment.svg?branch=master)](https://travis-ci.org/mgbellemare/Arcade-Learning-Environment)

<img align="right" src="doc/manual/figures/ale.gif" width=50>


### Arcade-Learning-Environment: An Evaluation Platform for General Agents

The Arcade Learning Environment (ALE) -- a platform for AI research.


This is the 0.5 release of the Arcade Learning Environment (ALE), a platform 
designed for AI research. ALE is based on Stella, an Atari 2600 VCS emulator. 
More information and ALE-related publications can be found at

http://www.arcadelearningenvironment.org

We encourage you to use the Arcade Learning Environment in your research. In
return, we would appreciate if you cited ALE in publications that rely on
it (BibTeX entry at the end of this document).

Feedback and suggestions are welcome and may be addressed to any active member 
of the ALE team.

Enjoy,
The ALE team

===============================
Quick start
===============================

Install main dependences:
```
sudo apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake
```

Compilation:

```
$ mkdir build && cd build
$ cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON ..
$ make -j 4
```

To install python module:

```
$ pip install .
or
$ pip install --user .
```

Getting the ALE to work on Visual Studio requires a bit of extra wrangling. You may wish to use IslandMan93's [Visual Studio port of the ALE.](https://github.com/Islandman93/Arcade-Learning-Environment)

For more details and installation instructions, see the [website](http://www.arcadelearningenvironment.org) and [manual](doc/manual/manual.pdf). To ask questions and discuss, please join the [ALE-users group](https://groups.google.com/forum/#!forum/arcade-learning-environment).


===============================
List of command-line parameters
===============================

Execute ./ale -help for more details; alternatively, see documentation 
available at http://www.arcadelearningenvironment.org.

```
-random_seed [n] -- sets the random seed; defaults to the current time

-game_controller [fifo|fifo_named] -- specifies how agents interact
  with ALE; see Java agent documentation for details

-config [file] -- specifies a configuration file, from which additional 
  parameters are read

-run_length_encoding [false|true] -- determine whether run-length encoding is
  used to send data over pipes; irrelevant when an internal agent is 
  being used

-max_num_frames_per_episode [n] -- sets the maximum number of frames per
  episode. Once this number is reached, a new episode will start. Currently
  implemented for all agents when using pipes (fifo/fifo_named) 

-max_num_frames [n] -- sets the maximum number of frames (independent of how 
  many episodes are played)
```

=====================================
Citing The Arcade Learning Environment
=====================================

If you use ALE in your research, we ask that you please cite the following.

M. G. Bellemare, Y. Naddaf, J. Veness and M. Bowling. The Arcade Learning Environment: An Evaluation Platform for General Agents, Journal of Artificial Intelligence Research, Volume 47, pages 253-279, 2013.

In BibTeX format:

```
@ARTICLE{bellemare13arcade,
  author = {{Bellemare}, M.~G. and {Naddaf}, Y. and {Veness}, J. and {Bowling}, M.},
  title = {The Arcade Learning Environment: An Evaluation Platform for General Agents},
  journal = {Journal of Artificial Intelligence Research},
  year = "2013",
  month = "jun",
  volume = "47",
  pages = "253--279",
}
```


