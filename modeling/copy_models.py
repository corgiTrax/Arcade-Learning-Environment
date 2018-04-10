import sys
import shutil

game_names = ['Breakout', 'Centipede', 'Enduro', 'Freeway', 'Mspacman', 'Riverraid', 'Seaquest', 'Venture']

if len(sys.argv) < 4:
	print("Usage: %s target_dir new_fname_extension(.hdf5|.gazeIsFrom.Img+OF.hdf5|...) hdf5_1 hdf5_2 ..." % sys.argv[0])
	print("Note that hdf5_x files needs to have game name (8 games now) in the path. " )
	sys.exit(1)

target_dir = sys.argv[1]
new_fname_extension = sys.argv[2]

src_files = []
for f in sys.argv[3:]:
	if f.endswith('model.hdf5'):
		#identify which game this file is
		this_game = ''
		for gname in game_names:
			if gname.lower() in f.lower():
				this_game = gname.lower()
				break
		if this_game == '':
			print("Directory path does not contain the game name!")
			sys.exit(1)

		new_fname = this_game + new_fname_extension
		dst = target_dir + new_fname
		command = "cp " + f + ' ' + dst
		raw_input("Executing command: %s \n Confirm? Ctrl-C to exit" % command)
		shutil.copy(f, dst)
	else:
		print("%s is not a model.hdf5 file!" % f)	






