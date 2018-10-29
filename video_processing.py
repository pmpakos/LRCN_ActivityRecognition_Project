from image_processing import *
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import to_categorical
from PIL import Image
import glob
import random 
import threading

seq_len = 16
crop_width = 227
crop_height = 227
num_labels = 101
iterations = 10 #40
stride = 8


def read_video_random(videoname,pseudoseed,data_type):    
    files = sorted(glob.glob(videoname+'/*.jpg'))
    tot_files = len(files)
    
    a = pseudoseed*(tot_files - seq_len)//iterations
    b = (pseudoseed+1)*(tot_files - seq_len)//iterations
    random_start = random.randint(a=a,b=b)
    files = files[random_start : random_start + seq_len]

    video = np.zeros((len(files),crop_width,crop_height,3))
    for i in range(len(files)):
        im=Image.open(files[i])
        im=im.resize((320,240),Image.ANTIALIAS)
        im=np.array(im,dtype=np.float64)
        if(data_type=='flow_images'):
            im = preprocess_input_flow(im)
        elif(data_type=='frames'):
            im = preprocess_input(im)
        im = im*1./255
        video[i,:,:,:] = center_crop(im,crop_width,crop_height)
    return video


class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen
        
@threadsafe_generator
def frame_generator(filenames,batch_size,data_type,pseudoseed):
    path_to_images = '/home/pasok/Desktop/pmpakos2/full_videos_dataset/'+data_type+'/'
    while 1:
        # Generate batch_size samples.
        batch = random.sample(filenames, batch_size)
        # pseudoseed = random.randint(a=0,b=iterations-1)
#         print("Creating generator with samples pseudoseed =",pseudoseed)
#         for i in batch:
#             print(i.split('\n')[0])
        batch_videos = np.zeros((batch_size , seq_len, crop_width, crop_height, 3))
        batch_labels = np.zeros((batch_size ,num_labels))
        for j in range(batch_size):
            # Reset to be safe.
            videoname = path_to_images + batch[j].split()[0].split('/')[1]
            label = int(batch[j].split()[1])
            batch_labels[j, label ] = 1
            batch_videos[j,:,:,:,:] = read_video_random(videoname,pseudoseed,data_type)
        yield batch_videos, batch_labels

###########################################################################################
def read_test_video_part(files,data_type):
    video = np.zeros((1,seq_len,crop_width,crop_height,3))
    for i in range(seq_len):
        im=Image.open(files[i])
        im=im.resize((320,240),Image.ANTIALIAS)
        im=np.array(im,dtype=np.float64)
        if(data_type=='frames'):
            im = preprocess_input(im)
        elif(data_type=='flow_images'):
            im = preprocess_input_flow(im)
        im = im*1./255
        video[0,i,:,:,:] = center_crop(im,crop_width,crop_height)
    return video

def get_video_pred(videoname,model,data_type):
    files = sorted(glob.glob(videoname+'/*.jpg'))
    tot_files = len(files)
    preds = np.zeros(((tot_files-seq_len)//stride,num_labels))
    for sp in range((tot_files-seq_len)//stride):
        video = read_test_video_part(files[sp*stride:sp*stride+seq_len],data_type)
        preds[sp,:] = np.mean(model.predict(video),axis=0)
    pred = np.mean(preds,axis=0)
    return pred




def read_video(videoname,data_type):
    files = sorted(glob.glob(videoname+'/*.jpg'))
    video = np.zeros((len(files),crop_width,crop_height,3))
    for i in range(len(files)):
        im=Image.open(files[i])
        im=im.resize((320,240),Image.ANTIALIAS)
        im=np.array(im,dtype=np.float64)
        if(data_type=='flow'):
            im = preprocess_input_flow(im)
        else:
            im = preprocess_input(im)
        im = im*1./255
        video[i,:,:,:] = center_crop(im,crop_width,crop_height)
    return video

# def generate_sequences(video,label,phase,si):
#     video = video.reshape(video.shape[0],crop_width*crop_height*3)
#     labels = np.ones((video.shape[0]))*int(label)
#     if(phase=='train'):
#         stride = 16
#         start_index = si
#     else:
#         stride=8
#         start_index = si

#     generator = TimeseriesGenerator(data=video,
#                                     targets=labels,
#                                     length=seq_len,
#                                     sampling_rate=1,
#                                     stride=stride,
#                                     start_index=start_index,
#                                     end_index=None,
#                                     shuffle=False,
#                                     reverse=False,
#                                     batch_size=1)
#     return generator

# def get_video_sequences(gen):
#     generated_sequences = np.zeros((len(gen),seq_len,crop_width,crop_height,3))
#     labels2 = np.zeros((len(gen)))
#     for i in range(len(gen)):
#         flat_seq = gen[i][0]
#         labels2[i]  = gen[i][1]
#         seq = flat_seq.reshape(seq_len,crop_width,crop_height,3)
#         generated_sequences[i,:,:,:] = seq

#     labels = to_categorical(labels2,num_classes=num_labels)
#     return generated_sequences,labels