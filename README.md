# Latent-DSB

## Quick Start

### Initialize the project

```bash
uvx invoke build
```

### Fill out `.env`

Add relevant API keys and paths (local data path, Weights & Biases, Hugging Face, etc.).

### (Optional) Install pre-commit

```bash
pip install pre-commit
```

## Running Experiments

- Run an experiment:

```bash
python main.py experiment=<experiment-name>
```

- Override arguments on the command line:

```bash
python main.py experiment=<experiment-name> data.batch_size=2
```

- Run with compiled model:

```bash
python main.py experiment=<experiment-name> compile=True
```

- Continue a previous training run using W&B run ID + checkpoint file:

```bash
python main.py experiment=<experiment-name> continue_from_id=<id> ckpt_filepath=<filepath>
```

- Start a new run but initialize from another checkpoint:

```bash
python main.py.py experiment=<experiment-name> ckpt_filepath=<filepath>
```

## Invoke Tasks

Run tasks with:

```bash
invoke <task-name>
```

Available tasks are defined in `tasks.py`. For example:

```bash
invoke format
```

## Are You Missing Anything?

Potentially useful additions:

- Authenticate external services if needed (for example, `wandb login`).
- Mention the expected `.env` keys (`DATA_PATH`, `WANDB_ENTITY`, `WANDB_API_KEY`, `HF_TOKEN`, etc.).
- Add a short troubleshooting section for missing env vars, path issues, or CUDA/GPU setup.
