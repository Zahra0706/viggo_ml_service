
import zipfile
from pathlib import Path

# Remplace par le chemin de ton archive de 5 Go
ZIP_PATH = r"C:\Users\hp\Downloads\archive.zip"

OUTPUT = Path("samples/session2")
SESSION = "dmd_frames/gB/10/s2/"
VIEWS = ["body", "face", "hands"]

with zipfile.ZipFile(ZIP_PATH, "r") as archive:
    for view in VIEWS:
        folder = f"frames_rgb_{view}"
        prefix = SESSION + folder + "/"

        images = sorted(
            name for name in archive.namelist()
            if name.startswith(prefix)
            and name.lower().endswith(".jpg")
        )

        selected = images[:100]

        destination = OUTPUT / folder
        destination.mkdir(parents=True, exist_ok=True)

        for name in selected:
            target = destination / Path(name).name
            with archive.open(name) as source:
                with open(target, "wb") as output:
                    import shutil
                    shutil.copyfileobj(source, output)

        print(f"{folder}: {len(selected)} images extraites")

print("Extraction terminée !")
