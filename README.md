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

