
AI playing 
=================


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
