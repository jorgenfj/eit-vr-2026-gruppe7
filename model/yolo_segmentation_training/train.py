import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

# Load environment variables from .env file
load_dotenv()

# Print CUDA info
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch cuDNN version: {torch.backends.cudnn.version()}")

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Parameters
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
PROJECT_ID = config["project_id"]
WORKSPACE_ID = config["workspace_id"]
VERSION = config["version"]
MODEL_TYPE = config.get("model_type")
EPOCHS = config.get("epochs")
PATIENCE = config.get("patience")
IMGSZ = config.get("imgsz")
BATCH = config.get("batch")
RESULTS_DIR = config.get("results_dir")
EXPORT_FORMATS = config.get("export_formats")
DATASET_FORMAT = config.get("dataset_format")

if not ROBOFLOW_API_KEY:
    raise ValueError("ROBOFLOW_API_KEY not set in environment or .env file.")


# Download dataset from Roboflow into 'roboflow_data/'
roboflow_data_dir = Path(__file__).parent / "roboflow_data"
dataset_dir = roboflow_data_dir / f"{PROJECT_ID}-v{VERSION}"
dataset_dir.mkdir(parents=True, exist_ok=True)
force_re_download = False

if not any(dataset_dir.iterdir()) or force_re_download:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
    version = project.version(VERSION)

    dataset = version.download(
        DATASET_FORMAT, location=str(dataset_dir), overwrite=True
    )
else:
    print("Dataset already exists. Skipping download.")

# Set up training configuration
model = YOLO(MODEL_TYPE)
data_yaml_path = dataset_dir / "data.yaml"

# Select device: CUDA, MPS (Apple Silicon), or CPU
if torch.cuda.is_available():
    device = 0
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = 'mps'
    print("Using Apple Silicon MPS device.")
else:
    device = 'cpu'
    print("Using CPU.")

slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
if slurm_cpus:
    WORKERS = int(slurm_cpus)
    print(f"Running on Cluster: Using {WORKERS} Slurm-allocated workers")
else:
    local_cpus = os.cpu_count() or 4
    WORKERS = min(8, local_cpus // 2)
    print(f"Running Locally: Using {WORKERS} out of {local_cpus} available CPU cores")

model_base = Path(MODEL_TYPE).stem
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
run_name = f"{model_base}-{dataset_dir.name}-{timestamp}".replace(" ", "-")

print(f"Training run name: {run_name}")

model.train(
    data=str(data_yaml_path),
    epochs=EPOCHS,
    patience=PATIENCE,
    imgsz=IMGSZ,
    batch=BATCH,
    device=device,
    project=str(RESULTS_DIR),
    name=run_name,
    workers=WORKERS,
    # compile=True
)


# Evaluate the model
metrics = model.val()
print(f"Validation metrics: {metrics}")


# Export the trained model
for fmt in EXPORT_FORMATS:
    model.export(format=fmt, device=device)
