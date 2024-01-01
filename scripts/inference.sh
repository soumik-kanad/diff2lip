#!/bin/bash

#set paths and arguments
real_video_root='dataset/VoxCeleb2/vox2_test_mp4/mp4/'
model_path="checkpoints/checkpoint.pt"
sample_path="output_dir"
sample_mode="cross" # or "reconstruction"
NUM_GPUS=2





#cross vs reconstruction
filelist_recon='dataset/filelists/voxceleb2_test_n_5000_reconstruction_5k.txt'
filelist_cross='dataset/filelists/voxceleb2_test_n_5000_seed_797_cross_5K.txt' 
if [ "$sample_mode" = "reconstruction" ]; then
    sample_input_flags="--sampling_input_type=first_frame --sampling_ref_type=first_frame"
    filelist=$filelist_recon
elif [ "$sample_mode" = "cross" ]; then
    sample_input_flags="--sampling_input_type=gt --sampling_ref_type=gt"
    filelist=$filelist_cross
else
    echo "Error: sample_mode can only be \"cross\" or \"reconstruction\""
    exit 0
fi
test_video_dir=$real_video_root
mkdir -p $sample_path
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --learn_sigma True --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm False"
DIFFUSION_FLAGS="--predict_xstart False  --diffusion_steps 1000 --noise_schedule linear --rescale_timesteps False"
SAMPLE_FLAGS="--sampling_seed=7   $sample_input_flags --timestep_respacing ddim25 --use_ddim True --model_path=$model_path --sample_path=$sample_path"
DATA_FLAGS="--nframes 5 --nrefer 1 --image_size 128 --sampling_batch_size=32 "
TFG_FLAGS="--face_hide_percentage 0.5 --use_ref=True --use_audio=True --audio_as_style=True"
GEN_FLAGS="--generate_from_filelist 1 --test_video_dir=$test_video_dir --filelist=$filelist --save_orig=False --face_det_batch_size 64 --pads 0,0,0,0"

if  [ "$NUM_GPUS" -gt 1 ]; then 
    mpiexec -n $NUM_GPUS python generate_dist.py $MODEL_FLAGS  $DIFFUSION_FLAGS  $SAMPLE_FLAGS $DATA_FLAGS $TFG_FLAGS $GEN_FLAGS
else
    python generate.py $MODEL_FLAGS  $DIFFUSION_FLAGS  $SAMPLE_FLAGS $DATA_FLAGS $TFG_FLAGS $GEN_FLAGS
fi
