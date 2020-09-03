To build the applications, enter in the terminal shell of your machine:

```
cd balltrajectorysd/singularity
./build_final_image.sh
```

Once this build is complete, you will find a new file in the directory with a naming convention of

```
final_balltrajectorysd_YYYY-MM-DD_HH_MM_SS.sif
```


A sample command for the execution of an experiment using the VAE is given by,
```
./final_balltrajectorysd_YYYY-MM-DD_HH_MM_SS.sif balltrajectorysd_vae \
    --number-gen=6001 \
    --number-cpus=-1 \
    --pct-random=0.6 \
    --full-loss=true \
    --pct-extension=0 \
    --loss-func=2 \
    --sample-train=false \
    --sample=false \
    --sne=2 
```

The AE implementation can be achieved by setting

```
    --full-loss=true
    --sample-train=false
    --sample=false
    --sne=0
```

The AURORA implementation can be executed using ```balltrajectorysd_aurora``` as the second argument. For instance,

```
./final_balltrajectorysd_YYYY-MM-DD_HH_MM_SS.sif balltrajectorysd_aurora \
    --number-gen=6001 \
    --number-cpus=-1 \
    --pct-random=0.6 \
    --full-loss=false \
    --pct-extension=0 \
    --loss-func=2 \
    --sample-train=false \
    --sample=false \
    --sne=0 
```

The applications offer GPU support. This requires a CUDA-enabled GPU on the machine and a CUDA driver version of 10.1.
By replacing ```./final_balltrajectorysd_YYYY-MM-DD_HH_MM_SS.sif``` with ```singularity run --nv final_balltrajectorysd_YYYY-MM-DD_HH_MM_SS.sif``` in the example commands, the applications will automatically detect a compatible GPU and use it for the Neural Network operations.

