import json
import numpy as np
import onnxruntime as ort
from pathlib import Path

BASE = Path(__file__).parent
MODEL = BASE / "models/swin/driver_swin.onnx"
PARITY_INPUT = BASE / "models/swin/parity_input_float32.bin"
PARITY_EXPECTED = BASE / "models/swin/parity_expected.json"
LABELS = BASE / "models/swin/labels.json"

parity_expected = json.loads(PARITY_EXPECTED.read_text())
labels = json.loads(LABELS.read_text())

arr = np.frombuffer(PARITY_INPUT.read_bytes(), dtype=np.float32).reshape(1, 3, 224, 224)

options = ort.SessionOptions()
options.intra_op_num_threads = 2
session = ort.InferenceSession(str(MODEL), sess_options=options, providers=["CPUExecutionProvider"])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"[INFO] Input name  : {input_name}")
print(f"[INFO] Output name : {output_name}")
print(f"[INFO] Input shape  : {session.get_inputs()[0].shape}")
print(f"[INFO] Output shape : {session.get_outputs()[0].shape}")

logits = session.run([output_name], {input_name: arr})[0][0]
expected_logits = np.array(parity_expected["logits"][0])

print(f"\n[RESULT] Logits produits : {logits.round(4).tolist()}")
print(f"[RESULT] Logits attendus : {expected_logits.round(4).tolist()}")

diff = np.abs(logits - expected_logits)
print(f"[RESULT] Diff max absolue : {diff.max():.6f}")
print(f"[RESULT] Parité OK (tol 1e-3) : {diff.max() < 1e-3}")

predicted_idx = int(np.argmax(logits))
expected_idx = parity_expected["class_index"]
print(f"[RESULT] Classe prédite : {predicted_idx} ({labels[predicted_idx]})")
print(f"[RESULT] Classe attendue : {expected_idx} ({labels[expected_idx]})")
print(f"[RESULT] Classe correcte : {predicted_idx == expected_idx}")

# Softmax
exp_l = np.exp(logits - logits.max())
scores = exp_l / exp_l.sum()
print(f"\n[SCORES SOFTMAX]")
for i, lbl in enumerate(labels):
    print(f"  {lbl:<20} : {scores[i]:.5f}")
