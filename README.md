# README

## Setup
Before running experiments, ensure your environment is set up correctly:

1. Run the setup script:
   ```bash
   python setup.py
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Experiments
To start an experiment, use the following command:
```bash
python train.py +experiment=experiment_name
```

### Available Experiments
You can find the different experiment configurations in the `configs/experiment/` folder.

## Overwriting Hyperparameters
You can override hyperparameters directly in the command line. For example, to change the batch size:
```bash
python train.py +experiment=experiment_name data.batch_size=32
```

## Running Experiments on Remote Machines
Experiments can be run on remote machines using the scripts in the `bashfiles` folder. To run an experiment remotely, use the following command:
```bash
bsub < bashfiles/experiment.sh
```
where `experiment.sh` is the filename containing the experiment bashfile.

## Project TODO List

- [ ] Make plot of "uncertainty" in generated sample vs. SDR. What do we expect?
- [ ] Compare GFB with DSB (STFT version).
   - How do they compare on key metrics?
   - Compare curvature displacement and total length of trajectories.
   - Compare both models to baseline model (i.e. no sampling)
- [x] Make plot of key metrics vs. noise factor. What is the best value for the noise factor?
- [x] Make plot of key metrics vs. number of steps.
- [ ] Make plot of key metrics vs. number of steps with deterministic sampling. 
- [ ] Make plot of key metrics vs. DSB iteration. 
- [x] Make plot of curvature displacement vs DSB iteration