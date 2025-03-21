import distillmos
import torch
from torch import Tensor

class MOS:
    def __init__(self, device = 'cuda'):
        sqa_model = distillmos.ConvTransformerSQAModel().to(device)
        sqa_model.eval()
        self.sqa_model = sqa_model
        self.device = device
    
    @torch.no_grad()
    def evaluate(self, samples : Tensor) -> float:
        samples = samples.to(self.device)
        if samples.dim() == 3:
            samples = samples.mean(dim=1)
        mos : Tensor = self.sqa_model(samples)
        mos = mos.mean().item()
        return mos
    
class KAD:
    def __init__(self, alpha : float = 100.):
        self.alpha = alpha
    
    @staticmethod
    def kernel(x : Tensor, y : Tensor) -> Tensor:
        return torch.exp(-torch.norm(x - y) ** 2)
    
    @torch.no_grad()
    def evaluate(self, generated : Tensor, real : Tensor) -> float:
        assert generated.shape[1:] == real.shape[1:], "The generated and real tensors must have the same shape."
        assert generated.device == real.device, "The generated and real tensors must be on the same device."
        
        n = generated.size(0)
        m = real.size(0)
        
        Kxx = torch.zeros(n, n)
        Kyy = torch.zeros(m, m)
        Kxy = torch.zeros(n, m)
        
        for i in range(n):
            for j in range(n):
                Kxx[i, j] = self.kernel(generated[i], generated[j])
                
        for i in range(m):
            for j in range(m):
                Kyy[i, j] = self.kernel(real[i], real[j])
                
        for i in range(n):
            for j in range(m):
                Kxy[i, j] = self.kernel(generated[i], real[j])
                
        Kxx = Kxx.sum() / (n * (n - 1))
        Kyy = Kyy.sum() / (m * (m - 1))
        Kxy = Kxy.sum() / (n * m)
        
        return self.alpha * (Kxx + Kyy - 2 * Kxy).item()
    
if __name__ == "__main__":
    x1 = torch.randn(16, 1, 16000)
    x2 = torch.randn(16, 1, 16000)
    x3 = x1 + 0.01 * torch.randn(16, 1, 16000)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    mos = MOS(device)
    kad = KAD()
    
    print(f"MOS: {mos.evaluate(x1)}")
    print(f"KAD (high): {kad.evaluate(x1, x2)}")
    print(f"KAD (low): {kad.evaluate(x1, x3)}")