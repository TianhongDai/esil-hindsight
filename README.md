# Episodic Self-Imitation Learning with Hindsight (ESIL)
This is the official code of our paper "Episodic Self-Imitation Learning with Hindsight".

## Requirements
- python=3.6.8
- pytorch=1.0.1
- mujoco-py=1.50.1.56

## Instructions
- run the **FetchReach-v1**:
```bash
mpirun -np 16 python train.py --env-name='FetchReach-v1' --adaptive-beta --display-interval=1 --total-frames=2500000

```
- run the **FetchPush-v1**:
```bash
mpirun -np 16 python train.py --env-name='FetchPush-v1' --adaptive-beta --display-interval=1 --total-frames=2500000

```
- run the **FetchPickAndPlace-v1**:
```bash
mpirun -np 32 python train.py --env-name='FetchPickAndPlace-v1' --adaptive-beta --display-interval=1 --batch-size=40 --ncycles=100 --total-frames=5000000

```
- run the **FetchSlide-v1**:
```bash
mpirun -np 32 python train.py --env-name='FetchSlide-v1' --adaptive-beta --display-interval=1 --batch-size=40 --ncycles=100 --total-frames=5000000

```
- Run the demo (e.g. **FetchPickAndPlace**):
```bash
python demo.py --env-name='FetchPickAndPlace-v1' --render

```
