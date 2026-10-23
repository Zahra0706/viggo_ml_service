"""Inférence sans entraînement. Exécuter uniquement avec checkpoints de provenance fiable."""
import argparse
import json
from pathlib import Path
import torch
from model_architecture import RGBDriverFusionNet, IRDDriverFusionNet
from preprocessing import load_streams

CLASSES = {
 "distraction": ["safe_drive","drinking","phonecall","texting","radio","reaching","talking_to_passenger","hair_and_makeup"],
 "drowsiness": ["eyes_open","eyes_closed","yawning"],
 "gaze": ["looking_road","not_looking_road"],
 "hands": ["both_hands","one_hand","no_hands"],
}

def load_model(architecture, checkpoint_path, device):
    model = architecture().to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or 'model_state_dict' not in ckpt:
        raise ValueError(f"Format de checkpoint inattendu: {checkpoint_path}")
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    return model.eval()

def predict(root, rgb_path, ird_path=None, alpha=0.7, device='cpu'):
    device = torch.device(device)
    if not 0 <= alpha <= 1: raise ValueError('alpha doit être compris entre 0 et 1')
    rgb = load_model(RGBDriverFusionNet, rgb_path, device)
    rgb_streams = {k: v.to(device) for k,v in load_streams(root,'rgb').items()}
    with torch.inference_mode():
        rgb_probs = {k: torch.softmax(v,dim=1) for k,v in rgb(rgb_streams).items()}
    if ird_path:
        ird = load_model(IRDDriverFusionNet, ird_path, device)
        ird_streams = {k: v.to(device) for k,v in load_streams(root,'ird').items()}
        with torch.inference_mode():
            ird_probs = {k: torch.softmax(v,dim=1) for k,v in ird(ird_streams).items()}
        probs = {k: alpha*rgb_probs[k]+(1-alpha)*ird_probs[k] for k in CLASSES}
        mode = 'fusion'
    else:
        probs = rgb_probs
        mode = 'rgb_only'
    return {'mode': mode, 'predictions': {task: {'class': CLASSES[task][int(p.argmax(dim=1).item())], 'class_index': int(p.argmax(dim=1).item()), 'scores': {label:round(float(p[0,i]),6) for i,label in enumerate(CLASSES[task])}} for task,p in probs.items()}}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames-root',required=True,help='Dossier contenant frames_rgb_body etc.')
    ap.add_argument('--rgb',default='models/rgb_best_model.pth')
    ap.add_argument('--ird',default=None,help='Si fourni, exécute la fusion RGB+IRD')
    ap.add_argument('--alpha',type=float,default=.7)
    ap.add_argument('--device',default='cpu')
    args=ap.parse_args()
    print(json.dumps(predict(args.frames_root,args.rgb,args.ird,args.alpha,args.device),indent=2,ensure_ascii=False))
