''' consistent initial noise for video generation'''
import cv2
import os
from os.path import join, basename, dirname, splitext
import shutil
import argparse
import numpy as np
import random
import torch, torchvision
import subprocess
from audio import audio
import face_detection
from tqdm import tqdm
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    tfg_model_and_diffusion_defaults,
    tfg_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from time import time
import torch.distributed as dist
from guided_diffusion.tfg_data_util import (
    tfg_process_batch,
) 

def get_frame_id(frame):
    return int(basename(frame).split('.')[0])

def crop_audio_window(spec, start_frame, args ):
    if type(start_frame) == int:
        start_frame_num = start_frame
    else:
        start_frame_num = get_frame_id(start_frame)
    start_idx = int(args.mel_steps_per_sec * (start_frame_num / float(args.video_fps)))
    end_idx = start_idx + args.syncnet_mel_step_size
    return spec[start_idx : end_idx, :]

def load_all_indiv_mels(path, args):
    in_path = path
    out_dir = join(args.sample_path, "temp",str(dist.get_rank()), basename(in_path).replace(".mp4", ""))
    os.makedirs(out_dir, exist_ok= True)
    out_path = join(out_dir, "audio.wav")
    command2 = 'ffmpeg -loglevel error -y -i {} -strict -2 {}'.format(in_path, out_path)
    subprocess.call(command2, shell=True)
    wav = audio.load_wav(out_path, args.sample_rate)
    orig_mel = audio.melspectrogram(wav).T

    all_indiv_mels = []
    # i=0
    i=1
    while True:
        m = crop_audio_window(orig_mel.copy(), max(i - args.syncnet_T//2,0), args)
        if (m.shape[0] != args.syncnet_mel_step_size):
            break
        all_indiv_mels.append(m.T)
        i+=1
    
    #clean up
    shutil.rmtree(join(args.sample_path, "temp", str(dist.get_rank())))
    
    return all_indiv_mels, wav

def load_video_frames(path, args):
    in_path = path
    out_dir = join(args.sample_path, "temp", str(dist.get_rank()), basename(in_path).replace(".mp4", ""), "image")
    os.makedirs(out_dir, exist_ok= True)


    command = "ffmpeg -loglevel error -y -i {} -vf fps={} -q:v 2 -qmin 1 {}/%05d.jpg".format(in_path, args.video_fps, out_dir)
    subprocess.call(command, shell=True)

    video_frames=[]
    for i, img_name in enumerate(sorted(os.listdir(out_dir))):
        img_path=join(out_dir, img_name)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        video_frames.append(img)

    #clean up
    shutil.rmtree(join(args.sample_path, "temp", str(dist.get_rank())))


    return video_frames


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def my_voxceleb2_crop(img):
    return img[:-int(img.shape[0]*2.36/8) , int(img.shape[1]*1.8/8): -int(img.shape[1]*1.8/8)]

def my_voxceleb2_crop_bboxs(img):
    return 0,img.shape[0]-int(img.shape[0]*2.36/8), int(img.shape[1]*1.8/8), img.shape[1]-int(img.shape[1]*1.8/8)

def face_detect(images, detector, args, resize=False):
    batch_size = args.face_det_batch_size
    
    while 1:
        predictions = []
        try:
            for i in range(0, len(images), batch_size):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError('Image too big to run face detection on GPU')
            batch_size //= 2
            args.face_det_batch_size = batch_size
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    if type(args.pads) == str :
        args.pads = [int(x) for x in args.pads.split(",")]
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            raise ValueError('Face not detected!')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = get_smoothened_boxes(np.array(results), T=5)
    
    if resize:
        if args.is_voxceleb2:
            results = [[cv2.resize(my_voxceleb2_crop(image),(args.image_size, args.image_size)), my_voxceleb2_crop_bboxs(image), True] for image, (x1, y1, x2, y2) in zip(images, boxes)]
        else:
            results = [[cv2.resize(image[y1: y2, x1:x2],(args.image_size, args.image_size)), (y1, y2, x1, x2), True] for image, (x1, y1, x2, y2) in zip(images, boxes)]
    else:
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2), True] for image, (x1, y1, x2, y2) in zip(images, boxes)]
    return results 

def normalise(tensor):
    """ [-1,1]->[0,1]"""
    return ((tensor+1)*0.5).clamp(0,1)

def normalise2(tensor):
    """ [0,1]->[-1,1]"""
    return (tensor*2-1).clamp(-1,1)


def sample_batch(batch, model, diffusion, args):
    B, F, C, H, W = batch[f'image'].shape
    sample_shape = (B*F, C, H, W)


    #generate fixed noise
    init_noise = None
    if args.sampling_seed:

        state = torch.get_rng_state()
        torch.manual_seed(args.sampling_seed)
        torch.cuda.manual_seed_all(args.sampling_seed)
        init_noise = torch.randn((1,C,H,W))
        #repeat noise for all frames
        init_noise = init_noise.repeat(B*F,1,1,1)
        torch.set_rng_state(state)


    img_batch, model_kwargs = tfg_process_batch(batch, args.face_hide_percentage, 
                                                use_ref=args.use_ref, 
                                                use_audio=args.use_audio, 
                                                # sampling_use_gt_for_ref=args.sampling_use_gt_for_ref, 
                                                noise=init_noise)
    

    img_batch = img_batch.to(dist_util.dev())
    model_kwargs = {k: v.to(dist_util.dev()) for k,v in model_kwargs.items()}
    init_noise = init_noise.to(dist_util.dev()) if init_noise is not None else None

    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    sample = sample_fn(
        model,
        sample_shape,
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        noise = init_noise
    )
    return sample, img_batch, model_kwargs


def generate(video_path, audio_path, model, diffusion, detector,  args, out_path=None, save_orig=True):
    video_frames = load_video_frames(video_path, args)
    try:
        face_det_results = face_detect(video_frames.copy(), detector, args, resize=True)
    except Exception as e:
        print("Error:", e, video_path, audio_path)
        import traceback
        print(traceback.format_exc())
    wrong_all_indiv_mels, wrong_audio_wavform = load_all_indiv_mels(audio_path, args)

    min_frames = min(len(video_frames), len(wrong_all_indiv_mels))
    video_frames = video_frames[:min_frames]
    face_det_results = face_det_results[:min_frames]
    face_bboxes = [face_det_results[i][1] for i in range(min_frames)]
    face_frames = torch.FloatTensor(np.transpose(np.asarray([face_det_results[i][0] for i in range(min_frames)], dtype=np.float32)/255.,(0,3,1,2)))#[N, C, H, W]
    wrong_all_indiv_mels = torch.FloatTensor(np.asarray(wrong_all_indiv_mels[:min_frames])).unsqueeze(1) #[N, 1, h, w]

    if save_orig:
        if out_path is None:
            out_path_orig = os.path.join(args.sample_path, splitext(basename(video_path))[0]+"_"+ splitext(basename(audio_path))[0]+"_orig.mp4") 
        else:
            out_path_orig = out_path.replace(".mp4", "_orig.mp4")
        torchvision.io.write_video(
            out_path_orig,
            video_array=torch.from_numpy(np.array(video_frames)), fps = args.video_fps, video_codec='libx264',
            audio_array=torch.from_numpy(wrong_audio_wavform).unsqueeze(0), audio_fps=args.sample_rate, audio_codec='aac'
        ) 

    if args.sampling_ref_type=='gt':
        ref_frames = face_frames.clone()
    elif args.sampling_ref_type=='first_frame':
        ref_frames = face_frames[0:1].repeat(len(face_frames),1,1,1)
    elif args.sampling_ref_type=='random':
        rand_idx = random.Random(args.sampling_seed).randint(0, len(face_frames)-1)
        ref_frames = face_frames[rand_idx:rand_idx+1].repeat(len(face_frames),1,1,1)

    if args.sampling_input_type=='first_frame':
        face_frames = face_frames[0:1].repeat(len(face_frames),1,1,1)
        video_frames = np.array(video_frames[0:1]*len(video_frames))
        face_bboxes = np.array(face_bboxes[0:1]*len(face_bboxes))


    rank = dist.get_rank() 
    world_size = dist.get_world_size()
    chunk_size = int(np.ceil(min_frames/world_size))
    start_idx = rank * chunk_size
    end_idx = min(start_idx + chunk_size, min_frames)
    generated_video_frames = []
    b_s = args.sampling_batch_size

    # print(rank,"/",world_size, "chunk: [",start_idx,"-", end_idx,"/",min_frames,"]")

    dist.barrier()
    torch.cuda.synchronize()
    t1=time()
    # for i in range(0,min_frames, b_s*args.nframes):
    for i in range(start_idx,end_idx, b_s*args.nframes):
        slice_end = min(i+b_s*args.nframes, end_idx)
        # if rank==0:
        #     print("rank 0: slice:",i,":",slice_end)
        video_frames_batch = video_frames[i:slice_end]
        face_bboxes_batch = face_bboxes[i:slice_end]

        # try:
        if  (slice_end-i) % args.nframes==0:
            img_batch = face_frames[i:slice_end] #[BF, C, H, W]  
            img_batch = img_batch.reshape(-1, args.nframes, img_batch.size(-3), img_batch.size(-2), img_batch.size(-1))
            ref_batch = ref_frames[i:slice_end]
            ref_batch = ref_batch.reshape(-1, args.nframes, ref_batch.size(-3), ref_batch.size(-2), ref_batch.size(-1))
            wrong_indiv_mel_batch = wrong_all_indiv_mels[i:slice_end] #[BF, 1, h, w]
            wrong_indiv_mel_batch = wrong_indiv_mel_batch.reshape(-1, args.nframes, wrong_indiv_mel_batch.size(-3),wrong_indiv_mel_batch.size(-2),wrong_indiv_mel_batch.size(-1))
        # except: 
        else: # of the last batch, if B*F % nframes!=0, then the above reshape throws error
            # but internally everything is going to get converted to BF
            # ie. (B,F, C, H, W) -> (B*F, C, H, W)  but (B*F, 1, C, H, W) -> (B*F, C, H, W) 
            img_batch = face_frames[i:slice_end] #[BF, C, H, W]
            img_batch = img_batch.reshape(-1, 1, img_batch.size(-3), img_batch.size(-2), img_batch.size(-1))
            ref_batch = ref_frames[i:slice_end]
            ref_batch = ref_batch.reshape(-1, 1, ref_batch.size(-3), ref_batch.size(-2), ref_batch.size(-1))
            wrong_indiv_mel_batch = wrong_all_indiv_mels[i:slice_end] #[BF, 1, h, w]
            wrong_indiv_mel_batch = wrong_indiv_mel_batch.reshape(-1, 1, wrong_indiv_mel_batch.size(-3),wrong_indiv_mel_batch.size(-2),wrong_indiv_mel_batch.size(-1))
        

        batch = {"image":img_batch, 
                "ref_img":ref_batch, 
                "indiv_mels":wrong_indiv_mel_batch}   

        sample, img_batch, model_kwargs = sample_batch(batch, model, diffusion, args)   
        mask = model_kwargs['mask']
        recon_batch = sample * mask + (1. -mask)*img_batch #[BF, C, H, W]
        recon_batch = (normalise(recon_batch)*255).cpu().numpy().transpose(0,2,3,1) #[-1,1] -> [0,255]

        for g,v,b in zip(recon_batch, video_frames_batch, face_bboxes_batch):
            y1, y2, x1, x2 = b
            g = cv2.resize(g.astype(np.uint8), (x2 - x1, y2 - y1))            
            v[y1:y2, x1:x2] = g
            generated_video_frames.append(v)

    torch.cuda.synchronize()
    t3=time()
    all_generated_video_frames = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(all_generated_video_frames, generated_video_frames)  # gather not supported with NCCL
    all_generated_video_frames_combined = []
    [all_generated_video_frames_combined.extend(gvf) for gvf in all_generated_video_frames]
    generated_video_frames = all_generated_video_frames_combined
    
    torch.cuda.synchronize()
    t2=time()

    if dist.get_rank() == 0:
        print("Time taken for sampling, ", t2-t1, ",time without all  gather, ", t3-t1, ",frames/gpu, ", len(generated_video_frames), ",total frames, ", min_frames)
        print(wrong_audio_wavform.shape, np.array(generated_video_frames).shape)
        min_time = len(generated_video_frames)/args.video_fps # because video is already smaller because it got chopped accoding to the mel array length
        wrong_audio_wavform = wrong_audio_wavform[:int(min_time*args.sample_rate)]
        print(wrong_audio_wavform.shape, np.array(generated_video_frames).shape)
        if out_path is None:
            out_path = os.path.join(args.sample_path, splitext(basename(video_path))[0]+"_"+ splitext(basename(audio_path))[0]+".mp4") 
        torchvision.io.write_video(
            out_path,
            video_array=torch.from_numpy(np.array(generated_video_frames)), fps = args.video_fps, video_codec='libx264',
            audio_array=torch.from_numpy(wrong_audio_wavform).unsqueeze(0), audio_fps=args.sample_rate, audio_codec='aac'
        ) 
    dist.barrier()



              

def generate_from_filelist(test_video_dir, filelist, model, diffusion, detector,  args):
    video_names = []
    audio_names = []
    with open(filelist, "r") as f:
        lines = f.readlines()
    for line in tqdm(lines):
        try:
            audio_name, video_name = line.strip().split()
            audio_path = join(test_video_dir, audio_name+'.mp4')
            video_path = join(test_video_dir, video_name+'.mp4')
            out_path = join(args.sample_path,audio_name.replace('/','.')+"_"+video_name.replace('/','.')+".mp4")
            generate(video_path, audio_path, model, diffusion, detector,  args, out_path=out_path ,save_orig=args.save_orig)
        except Exception as e:
            print("Error:", e, video_path, audio_path)
            import traceback
            print(traceback.format_exc())

        

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure(dir=args.sample_path, format_strs=["stdout", "log"])

    logger.log("creating model...")
    model, diffusion = tfg_create_model_and_diffusion(
            **args_to_dict(args, tfg_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location='cpu')
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cuda' if torch.cuda.is_available() else 'cpu')

    if args.generate_from_filelist:
        generate_from_filelist(args.test_video_dir, args.filelist, model, diffusion, detector,  args)
    else:
        generate(args.video_path, args.audio_path, model, diffusion, detector,  args, out_path=args.out_path, save_orig=args.save_orig)


def create_argparser():
    defaults = dict(
        # generate from a single audio-video pair
        generate_from_filelist = False,
        video_path = "",
        audio_path = "",
        out_path = None, 
        save_orig = True,

        #generate from filelist : generate_from_filelist = True
        test_video_dir = "test_videos",
        filelist = "test_filelist.txt",


        use_fp16 = True,
        #tfg specific
        face_hide_percentage=0.5,
        use_ref=False,
        use_audio=False,
        audio_as_style=False,
        audio_as_style_encoder_mlp=False,
        
        #data args
        nframes=1,
        nrefer=0,
        image_size=128,
        syncnet_T = 5,
        syncnet_mel_step_size = 16,
        audio_frames_per_video = 16, #for tfg model, we use sound corresponding to 5 frames centred at that frame
        audio_dim=80,
        is_voxceleb2=True,

        video_fps=25,
        sample_rate=16000, #audio sampling rate
        mel_steps_per_sec=80.,

        #sampling args
        clip_denoised=True, # not used in training
        sampling_batch_size=2,
        use_ddim=False,
        model_path="",
        sample_path="d2l_gen",
        sample_partition="",
        sampling_seed=None,
        sampling_use_gt_for_ref=False,
        sampling_ref_type='gt', #one of ['gt', 'first_frame', 'random']
        sampling_input_type='gt', #one of ['gt', 'first_frame']
        
        # face detection args
        face_det_batch_size=64,
        pads = "0,0,0,0"
    )
    defaults.update(tfg_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__=="__main__":
    main()