# VIGGO — Premier prototype d’inférence hors ligne

Ce projet reprend les architectures **exactes** des cellules 12 et 14 du notebook fourni, avec un seul changement volontaire : `pretrained=False` pour empêcher un téléchargement des poids ImageNet. Les checkpoints fournis sont chargés ensuite via `load_state_dict(strict=True)`.

## Installation (Python 3.10+ recommandé)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Entrées attendues

Créez un dossier `samples/session1` contenant `frames_rgb_body`, `frames_rgb_face`, `frames_rgb_hands` (au moins 8 fichiers `frame_000000.jpg` etc. dans chacun). Pour la fusion, ajouter aussi `frames_ir_body`, `frames_ir_face`, `frames_ir_hands`, `frames_depth_body`, `frames_depth_face`, `frames_depth_hands`. Les neuf vues doivent être synchronisées. Idéalement utiliser des clips de 50 frames et tenir compte des décalages temporels des caméras présents dans le dataset source. Ici, le prototype suppose des frames déjà alignées et des séquences de mêmes durées.

Attention : la conversion RGB du notebook opère sur les images BGR décodées par OpenCV **sans** conversion supplémentaire BGR→RGB. Ce comportement est conservé ici pour compatibilité avec l'entraînement.

## Premier test

```bash
python test_inference.py --frames-root samples/session1 --mode rgb
python inference.py --frames-root samples/session1 --rgb models/rgb_best_model.pth
```

Avec les neuf flux synchronisés :

```bash
python test_inference.py --frames-root samples/session1 --mode fusion
python inference.py --frames-root samples/session1 --rgb models/rgb_best_model.pth --ird models/ird_best_model.pth --alpha 0.7
```

Le résultat JSON fournit quatre classifications et leurs scores. Les scores softmax ne doivent pas être assimilés à des probabilités cliniquement calibrées.

## État et limites

- Le code Python compile, mais aucun test de prédiction sur une vidéo réelle n’a été réalisé dans cet environnement.
- Il faut confirmer la compatibilité des poids et les versions PyTorch/timm.
- Les chemins des images sont fournis par l’utilisateur ; aucun dataset de frames n’est inclus.
- Le flux RGB seul est une alternative de laboratoire (trois vues RGB nécessaires), pas une implémentation à caméra unique sur smartphone.
- Les prédictions concernent **distraction/somnolence/regard/mains**, et non stress/FC/respiration/HRV.
- Ceci n'est pas un système de sécurité routière validé ; ne pas l’utiliser pour déclencher automatiquement un appel ou un SMS d’urgence.
