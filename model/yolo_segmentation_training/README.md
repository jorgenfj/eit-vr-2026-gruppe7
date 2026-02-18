# Object detection training

Scripts for training an image segmentation model using YOLOv8 on Roboflow datasets.

## Environment variables

Training requires a Roboflow API key.
Set it **once per shell session** before submitting the Slurm job or training locally:

```bash
export ROBOFLOW_API_KEY=<your_roboflow_api_key>
```

You can verify it is set with:

```bash
echo $ROBOFLOW_API_KEY
```

Alternatively create a .env file with the following content

```bash
ROBOFLOW_API_KEY=<your_roboflow_api_key>
```

**Do not commit your API key to Git.**

## Running the SLURM job

1. SSH into the cluster (IDUN)
2. Navigate to the project directory
3. Submit the job:

```bash
sbatch --export=ALL,ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY Job.slurm
```

After submission, Slurm will print a job ID.

## Monitoring the job

- Check job status:

```bash
squeue -j <JOB_ID>
```

- To list all your jobs:

```bash
squeue -u $USER
```

- Follow the job output:

```bash
tail -f $(ls -t vortex-instance-seg-train_*.out | head -n 1)
```

## Canceling a job

```bash
scancel <JOB_ID>
```
