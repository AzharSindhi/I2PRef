import torch
import sys
import os
sys.path.append(os.path.abspath("../"))
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist

cham_loss = chamfer_3DDist()


def fscore(dist1, dist2, threshold=0.001):
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    f = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    f[torch.isnan(f)] = 0
    return f, precision_1, precision_2


def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = cham_loss(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


def calc_cd(output, gt, calc_f1=False, return_raw=False):
    dist1, dist2, idx1, idx2 = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))
    cd_p = cd_p.mean()
    cd_t = cd_t.mean()
    res = [cd_p, cd_t]
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2)
        res.append(f1)
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    return res
