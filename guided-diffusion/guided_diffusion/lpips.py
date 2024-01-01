from lpips_pytorch import LPIPS
import torch

class LPIPS1(LPIPS):
    r"""
    Overrriding the LPIPS to send loss without reducing the batch
    Arguments:
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):
        super(LPIPS1, self).__init__(net_type = 'alex', version ='0.1')

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        feat_x, feat_y = self.net(x), self.net(y)
        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]
        # return torch.sum(torch.cat(res, 0), 0, True)
        return torch.sum(torch.cat(res, 1), 1, True)