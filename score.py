import torchaudio
import torch
import pdb
import os
import tqdm

from pesq import pesq
from pystoi import stoi
from torchmetrics import ScaleInvariantSignalNoiseRatio
cal_si_snr = ScaleInvariantSignalNoiseRatio()

sr = 16000
mode = 'wb'
# path = "./enhanced/"
path = "./separated/"

def get_path(root):
    paths = []
    for r, dirs, files in os.walk(root):
        for f in files:
            if f.endswith('.wav'):
                paths.append(os.path.join(r, f))
    return paths

def score(tgt, wav, noisy):
    pscore = pesq(sr, tgt, wav, mode)
    sscore = stoi(tgt, wav, sr)
    ssnr = cal_si_snr(torch.FloatTensor(wav), torch.FloatTensor(tgt))
    refp = pesq(sr, tgt, noisy, mode) 
    refs = stoi(tgt, noisy, sr)
    refsnr = cal_si_snr(torch.FloatTensor(noisy), torch.FloatTensor(tgt))

    return pscore, refp, ssnr, sscore, refs, refsnr 

def stat(lists, method, snr):
    pavg = 0
    savg = 0
    ssnravg = 0
    pref = 0
    sref = 0
    ssnrref = 0
    length = len(lists)
    # pdb.set_trace()
    print("\nprocessing", method, snr, "\n")
    for path in lists:
        wav = torchaudio.load(path)[0].squeeze().numpy()
        cpath = path.replace(method, "clean").replace(snr,"")
        npath = path.replace(method, "noisy") 
        cwav = torchaudio.load(cpath)[0].squeeze().numpy() 
        nwav = torchaudio.load(npath)[0].squeeze().numpy()
        #pdb.set_trace()
        pscore, refp, ssnr, sscore, refs, refsnr = score(cwav, wav, nwav) 
        pavg+=pscore/length
        savg+=sscore/length
        ssnravg+=ssnr/length
        pref+=refp/length
        sref+=refs/length
        ssnrref+=refsnr/length

    print("\n"+method, snr, "PESQ/STOI/SISNR/GTPESQ/GTSTOI/GTSISNR:  ", pavg, "/", savg, "/", ssnravg, "/",pref, "/",sref, "/", ssnrref, "\n")
        

if __name__=='__main__':
    ## SE
    # for method in ['final_project_AVSE_audio_only_stoiloss_speech_enhancement_version01-epoch=24-val_loss=0','final_project_AVSE_audio_visual_stoiloss_speech_enhancement_version_Pat-last','final_project_AVSE_audio_visual_stoiloss_speech_enhancement_version01-epoch=69-val_loss=0']:
    #     for snr in ['-5.0', '0.0', '5.0']:
    #         lists = get_path(path+method+"/"+snr)
    #         # pdb.set_trace()
    #         stat(lists, method, snr)
    
    ##SS
    for method in ['final_project_AVSE_audio_only_stoiloss_version01-epoch=99-val_loss=0.21','final_project_AVSE_audio_visual_stoiloss_version01-epoch=29-val_loss=0.18','final_project_AVSE_audio_visual_stoiloss_speech_separation_version_Pat-last']:
        for snr in ['0.0', '5.0', '10.0']:
            lists = get_path(path+method+"/"+snr)[-500:-200]
            # pdb.set_trace()
            stat(lists, method, snr)
