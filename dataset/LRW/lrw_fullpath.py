'''Converts the LRW video names in filelists to LRW relative paths and dumps them unto new filelists'''
import os
filelist = "../filelists/lrw_cross.txt" 

filelist_split_path = filelist.replace(".txt","_relative_path.txt")
with open(filelist, 'r') as f:
        lines = f.readlines()
with open(filelist_split_path, 'w') as f:
    for i in range(len(lines)):
        audio_name, video_name=lines[i].split(' ')
        audio_word = audio_name.split('_')[0]
        video_word = video_name.split('_')[0]
        f.write(os.path.join(audio_word,'test',audio_name)+' '+os.path.join(video_word,'test',video_name))

filelist = "../filelists/lrw_reconstruction.txt" 

filelist_split_path = filelist.replace(".txt","_relative_path.txt")
with open(filelist, 'r') as f:
        lines = f.readlines()
with open(filelist_split_path, 'w') as f:
    for i in range(len(lines)):
        audio_name, video_name=lines[i].split(' ')
        audio_word = audio_name.split('_')[0]
        video_word = video_name.split('_')[0]
        f.write(os.path.join(audio_word,'test',audio_name)+' '+os.path.join(video_word,'test',video_name))