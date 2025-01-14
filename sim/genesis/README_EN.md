# Genesis Simulation Environment - English Version

## Getting Started

### Navigate to genesis directory
```bash
cd sim/genesis
```

### Install rsl_rl
```bash
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl && git checkout v1.0.2 && pip install -e .
```

### Install TensorBoard
```bash
pip install tensorboard
```

## Training

After installation, start training by running:

```bash
python zeroth_train.py
```

To monitor the training process, start TensorBoard:

```bash
tensorboard --logdir logs
```

## Evaluation

To view training results:

```bash
python zeroth_eval.py
```

## Additional Notes

1. Make sure all dependencies are installed properly
2. Training logs will be saved in the `logs` directory
3. For advanced configuration, refer to the training script parameters
4. Ensure proper GPU setup if available for faster training
5. For Mac M-series chips, use micromamba instead of conda:
   ```bash
   micromamba activate genesis
   ```
   Then proceed with pip/python commands

## Troubleshooting

- If encountering dependency issues, try creating a virtual environment
- Check Python version compatibility (requires Python 3.8+)
- Verify CUDA installation if using GPU acceleration
