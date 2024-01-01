import torch

def normalise2(tensor):
    '''[0,1] -> [-1,1]'''
    return (tensor*2 - 1.).clamp(-1,1)

def tfg_data(dataloader, face_hide_percentage, use_ref, use_audio):#, sampling_use_gt_for_ref=False, noise = None):
    def inf_gen(generator):
        while True:
            yield from generator
    data = inf_gen(dataloader)
    for batch in data:
        img_batch, model_kwargs = tfg_process_batch(batch, face_hide_percentage, use_ref, use_audio)
        yield img_batch, model_kwargs
        

def tfg_process_batch(batch, face_hide_percentage, use_ref=False, use_audio=False, sampling_use_gt_for_ref=False, noise = None):
    model_kwargs = {}
    B, F,C, H, W = batch["image"].shape
    img_batch = normalise2(batch["image"].reshape(B*F, C, H, W).contiguous())
    model_kwargs = tfg_add_cond_inputs(img_batch, model_kwargs, face_hide_percentage, noise)
    if use_ref:
        model_kwargs = tfg_add_reference(batch, model_kwargs, sampling_use_gt_for_ref)
    if use_audio:
        model_kwargs = tfg_add_audio(batch,model_kwargs)
    return img_batch, model_kwargs

def tfg_add_reference(batch, model_kwargs, sampling_use_gt_for_ref=False):
    # assuming nrefer = 1
    #[B, nframes, C, H, W] -> #[B*nframes, C, H, W]
    if sampling_use_gt_for_ref:
        B, F,C, H, W = batch["image"].shape
        img_batch = normalise2(batch["image"].reshape(B*F, C, H, W).contiguous())
        model_kwargs["ref_img"] = img_batch
    else:
        _, _, C, H , W =  batch["ref_img"].shape
        ref_img = normalise2(batch["ref_img"].reshape(-1, C, H, W).contiguous())
        model_kwargs["ref_img"] = ref_img
    return model_kwargs

def tfg_add_audio(batch, model_kwargs):
    # unet needs [BF, h, w] as input
    B, F, _, h, w = batch["indiv_mels"].shape
    indiv_mels = batch["indiv_mels"] # [B, F, 1, h, w]
    indiv_mels = indiv_mels.squeeze(dim=2).reshape(B*F, h , w)
    model_kwargs["indiv_mels"] = indiv_mels
    # syncloss needs [B, 1, 80, 16] as input
    if "mel" in batch:
        mel = batch["mel"] #[B, 1, h, w]
        model_kwargs["mel"]=mel
    return model_kwargs

def tfg_add_cond_inputs(img_batch, model_kwargs, face_hide_percentage, noise=None):
    B, C, H, W = img_batch.shape
    mask = torch.zeros(B,1,H,W)
    mask_start_idx = int (H*(1-face_hide_percentage))
    mask[:,:,mask_start_idx:,:]=1.
    if noise is None:
        noise = torch.randn_like(img_batch)
    assert noise.shape == img_batch.shape, "Noise shape != Image shape"
    cond_img = img_batch *(1. - mask)+mask*noise

    model_kwargs["cond_img"] = cond_img
    model_kwargs["mask"] = mask
    return model_kwargs


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn=nn*s
        pp+=nn
    return pp