from src.data_modules import BaseDM
from src.datasets import DummyDataset
import pytest


def test_base_datamodule():
    trainsize, valsize, testsize = 100, 20, 20
    trainset = DummyDataset(size=trainsize)
    valset = DummyDataset(size=valsize)
    testset = DummyDataset(size=testsize)

    dm = BaseDM(trainset=trainset, train_val_split=0.8, batch_size=1)
    trainloader = dm.train_dataloader()
    valloader = dm.val_dataloader()
    testloader = dm.test_dataloader()

    assert testloader is None, "Test loader should be None when no testset is provided."
    assert len(trainloader.dataset) == trainsize * 0.8, "Train loader size mismatch."
    assert len(valloader.dataset) == trainsize * 0.2, "Validation loader size mismatch."

    dm = BaseDM(trainset, testset=testset, batch_size=1, train_val_split=0.8)
    trainloader = dm.train_dataloader()
    valloader = dm.val_dataloader()
    testloader = dm.test_dataloader()

    assert len(trainloader.dataset) == trainsize * 0.8, "Train loader size mismatch."
    assert len(valloader.dataset) == trainsize * 0.2, "Validation loader size mismatch."
    assert len(testloader.dataset) == testsize, "Test loader size mismatch."

    dm = BaseDM(trainset, valset=valset, testset=testset, batch_size=1)
    trainloader = dm.train_dataloader()
    valloader = dm.val_dataloader()
    testloader = dm.test_dataloader()

    assert len(trainloader.dataset) == trainsize, "Train loader size mismatch."
    assert len(valloader.dataset) == valsize, "Validation loader size mismatch."
    assert len(testloader.dataset) == testsize, "Test loader size mismatch."

    for batch in trainloader:
        assert "input" in batch, "Batch should have 'input' key."
        assert "target" in batch, "Batch should have 'target' key."

    for batch in valloader:
        assert "input" in batch, "Batch should have 'input' key."
        assert "target" in batch, "Batch should have 'target' key."

    for batch in testloader:
        assert "input" in batch, "Batch should have 'input' key."
        assert "target" in batch, "Batch should have 'target' key."


def test_base_datamodule_batch_sizes():
    trainset = DummyDataset(size=64)
    dm = BaseDM(trainset, batch_size=16, train_val_split=0.8)
    trainloader = dm.train_dataloader()
    batch = next(iter(trainloader))
    assert batch["input"].shape[0] == 16, "Batch size mismatch."


def test_base_datamodule_no_trainset():
    with pytest.raises((TypeError, ValueError)):
        BaseDM()
