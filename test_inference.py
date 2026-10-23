"""Test statique des données sans charger les checkpoints."""
import argparse
from pathlib import Path
from preprocessing import N_SEGMENTS
ap=argparse.ArgumentParser()
ap.add_argument('--frames-root',required=True)
ap.add_argument('--mode',choices=['rgb','fusion'],default='rgb')
args=ap.parse_args()
mods=['rgb'] if args.mode=='rgb' else ['rgb','ir','depth']
errors=[]
for mod in mods:
 for view in ['body','face','hands']:
    p=Path(args.frames_root)/f'frames_{mod}_{view}'
    count=len(list(p.glob('frame_*.jpg'))) if p.is_dir() else 0
    print(f'{p}: {count} images')
    if count<N_SEGMENTS: errors.append(str(p))
if errors: raise SystemExit('Flux absents ou trop courts: '+', '.join(errors))
print('OK: toutes les entrées requises existent. L’inférence reste à exécuter.')
