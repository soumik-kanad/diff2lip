#!/bin/bash

#set paths and arguments
sample_mode="cross" # or "reconstruction"
NUM_GPUS=1
generate_from_filelist=0
video_path="path/to/video.mp4"
audio_path="path/to/audio.mp4"
out_path="path/to/output.mp4"
model_path="path/to/model.pt"



#cross vs reconstruction 
if [ "$sample_mode" = "reconstruction" ]; then
    sample_input_flags="--sampling_input_type=first_frame --sampling_ref_type=first_frame"
elif [ "$sample_mode" = "cross" ]; then
    sample_input_flags="--sampling_input_type=gt --sampling_ref_type=gt"
else
    echo "Error: sample_mode can only be \"cross\" or \"reconstruction\""
    exit 0
fi

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --learn_sigma True --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm False"
DIFFUSION_FLAGS="--predict_xstart False  --diffusion_steps 1000 --noise_schedule linear --rescale_timesteps False"
SAMPLE_FLAGS="--sampling_seed=7   $sample_input_flags --timestep_respacing ddim25 --use_ddim True --model_path=$model_path"
DATA_FLAGS="--nframes 5 --nrefer 1 --image_size 128 --sampling_batch_size=32 "
TFG_FLAGS="--face_hide_percentage 0.5 --use_ref=True --use_audio=True --audio_as_style=True"
GEN_FLAGS="--generate_from_filelist $generate_from_filelist  --video_path=$video_path --audio_path=$audio_path --out_path=$out_path --save_orig=False --face_det_batch_size 64 --pads 0,0,0,0 --is_voxceleb2=False"

if  [ "$NUM_GPUS" -gt 1 ]; then 
    mpiexec -n $NUM_GPUS python generate_dist.py $MODEL_FLAGS  $DIFFUSION_FLAGS  $SAMPLE_FLAGS $DATA_FLAGS $TFG_FLAGS $GEN_FLAGS
else
    python generate.py $MODEL_FLAGS  $DIFFUSION_FLAGS  $SAMPLE_FLAGS $DATA_FLAGS $TFG_FLAGS $GEN_FLAGS
fi