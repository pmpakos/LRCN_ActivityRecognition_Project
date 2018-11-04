import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from custom_models import *
from image_processing import *
from inception_utils import *
from utils import *
from video_processing import *
import pickle

while True:
    try:
        # Note: Python 2.x users should use raw_input, the equivalent of 3.x's input
        aa = int(input("Please enter the S/N of the video to be checked [1-3783] or 0 for random choice : "))
    except ValueError:
        print("Sorry, I didn't understand that.")
        #better try again... Return to the start of the loop
        continue
    else:
        if aa>3783 or aa<0:
        	continue
        if aa==0:
            aa = random.randint(a=1,b=3783)
            break
        #aa was successfully parsed!
        #we're ready to exit the loop.
        break

test_file = open('data_a/ucf101_split1_testVideos.txt','r')
files = test_file.readlines()
test_file.close()

print("Video to be tested is :",files[aa-1].split()[0].split('/')[1],"\n")

root_path1 = 'full_videos_dataset/frames/'
root_path2 = 'full_videos_dataset/flow_images/'

videoname1 = root_path1 + files[aa-1].split()[0].split('/')[1]
video1 = read_video(videoname1,'frames')

videoname2 = root_path2 + files[aa-1].split()[0].split('/')[1]
video2 = read_video(videoname2,'flow_images')

classes_dictionary = pickle.load(open('data_a/action_dictionary.p','rb'))

input_shape = (crop_height,crop_width,3)
seq_input = (seq_len,crop_height,crop_width,3)

########################
rgb_p = 'saved_models/caffenet_single_rgb.hdf5'
model = CaffeDonahueFunctional(input_shape=input_shape,num_labels=num_labels)
model.load_weights(rgb_p)

rgb_pred = np.mean(model.predict(video1),axis=0)
caffenet_rgb_pred = 1.0*rgb_pred
print("caffenet_rgb classified video as",classes_dictionary[caffenet_rgb_pred.argmax()],"with score",caffenet_rgb_pred.max())
########################

########################
flow_p = 'saved_models/caffenet_single_flow.hdf5'
model_flow = CaffeDonahueFunctional(input_shape=input_shape,num_labels=num_labels)
model_flow.load_weights(flow_p)

flow_pred = np.mean(model_flow.predict(video2),axis=0)
caffenet_flow_pred = 1.0*flow_pred
print("caffenet_flow classified video as",classes_dictionary[caffenet_flow_pred.argmax()],"with score",caffenet_flow_pred.max())
########################

########################
rgb_p = 'saved_models/inception_rgb.hdf5'
model = get_model(weights='imagenet',input_shape=input_shape,num_labels=num_labels)
model.load_weights(rgb_p)

rgb_pred = np.mean(model.predict(video1),axis=0)
inception_rgb_pred = 1.0*rgb_pred
print("inception_rgb classified video as",classes_dictionary[inception_rgb_pred.argmax()],"with score",inception_rgb_pred.max())
########################


########################
rgb_p = 'saved_models/lstm_rgb.hdf5'
model = LSTMCaffeDonahueFunctional(seq_input=seq_input, num_labels=num_labels)
model.load_weights(rgb_p)

rgb_pred = get_video_pred(videoname1,model,'frames')
lstm_rgb_pred = 1.0*rgb_pred
print("lstm_rgb classified video as",classes_dictionary[lstm_rgb_pred.argmax()],"with score",lstm_rgb_pred.max())
########################


########################
flow_p = 'saved_models/lstm_flow.hdf5'
model_flow = LSTMCaffeDonahueFunctional(seq_input=seq_input, num_labels=num_labels)
model_flow.load_weights(flow_p)

flow_pred = get_video_pred(videoname2,model_flow,'flow_images')
lstm_flow_pred = 1.0*flow_pred
print("lstm_flow classified video as",classes_dictionary[lstm_flow_pred.argmax()],"with score",lstm_flow_pred.max())
########################


########################
flow_p = 'saved_models/lstm_flow_1024.hdf5'
model_flow = LSTMCaffeDonahueFunctional(seq_input=seq_input, num_labels=num_labels,check1024=True)
model_flow.load_weights(flow_p)

flow_pred = get_video_pred(videoname2,model_flow,'flow_images')
lstm_flow1024_pred = 1.0*flow_pred
print("lstm_flow_1024 classified video as",classes_dictionary[lstm_flow1024_pred.argmax()],"with score",lstm_flow1024_pred.max())
########################


########################
w = [0.33, 0.67]
weighted_pred = w[0]*caffenet_rgb_pred + w[1]*caffenet_flow_pred
print("1/3 caffenet_rgb & 2/3 caffenet_flow classified video as",classes_dictionary[weighted_pred.argmax()],"with score",weighted_pred.max())
########################


########################
w = [0.5, 0.5]
weighted_pred = w[0]*inception_rgb_pred + w[1]*caffenet_flow_pred
print("1/2 inception_rgb & 1/2 caffenet_flow classified video as",classes_dictionary[weighted_pred.argmax()],"with score",weighted_pred.max())
########################


########################
w = [0.33, 0.67]
weighted_pred = w[0]*lstm_rgb_pred + w[1]*lstm_flow_pred
print("1/3 lstm_rgb & 2/3 lstm_flow classified video as",classes_dictionary[weighted_pred.argmax()],"with score",weighted_pred.max())
########################


########################
w = [0.33, 0.67]
weighted_pred = w[0]*lstm_rgb_pred + w[1]*lstm_flow1024_pred
print("1/3 lstm_rgb & 2/3 lstm_flow(1024) classified video as",classes_dictionary[weighted_pred.argmax()],"with score",weighted_pred.max())
########################
