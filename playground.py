from src.dataset import AllVCTK

dataset = AllVCTK(train=True)
print(len(dataset))

dataset = AllVCTK(train=False)
print(len(dataset))