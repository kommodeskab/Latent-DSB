from src.lightning_modules.scheduler import DSBScheduler
import torch


def test_scheduler():
    scheduler = DSBScheduler(
        epsilon=1.0,
        target="drift",
        condition_on_start=False,
    )
    timeschedule = scheduler.get_timeschedule(num_steps=10, scheduler_type="linear")
    assert len(timeschedule) == 10, "Timeschedule should have 10 steps."

    for t_plus_one, t in timeschedule:
        assert 0 <= t_plus_one <= 1, "Timesteps should be in the range [0, 1]."
        assert t_plus_one >= t, "Timesteps should be non-decreasing."

    B, D = 4, 16
    x0_b = torch.randn(B, D)
    x1_b = torch.randn(B, D)
    x0_f = torch.randn(B, D)
    x1_f = torch.randn(B, D)

    batch = scheduler.sample_training_batch(
        x0_b=x0_b,
        x1_b=x1_b,
        x0_f=x0_f,
        x1_f=x1_f,
    )

    assert batch["xt"].shape == (
        2 * B,
        D,
    ), "Input batch should have shape (batch_size, 2 * feature_dim) when condition_on_start is False."
    assert batch["timesteps"].shape == (2 * B,), "Timesteps should have shape (batch_size,)."
    assert batch["conditional"].shape == (2 * B,), "Conditional should have shape (batch_size,)."
    assert batch["target"].shape == (2 * B, D), "Target should have shape (batch_size, feature_dim)."
