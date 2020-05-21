import human_gaze_predictor as hgp
import sys
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if len(sys.argv) < 4:
    print("Usage: %s gaze_model.hdf5 meanfile datafile" % sys.argv[0])
    sys.exit(1)

predictor = hgp.Human_Gaze_Predictor(sys.argv[1], sys.argv[2], sys.argv[3])
predictor.init_model()
predictor.init_data()
predictor.predict_and_save()


