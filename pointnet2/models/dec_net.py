import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import sample_farthest_points

class MappingNet(nn.Module):
    def __init__(self, K1,N=256):
        super(MappingNet, self).__init__()
        self.K1 = K1
        self.N=N

        self.fc1 = nn.Linear(self.N, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, self.K1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(self.K1)

    def forward(self, x):
        
        x = F.relu(self.bn1(self.fc1(x).permute(0,2,1)))
       
        x = x.permute(0,2,1)
        x = F.relu(self.bn2(self.fc2(x).permute(0,2,1)))
      
        x = x.permute(0,2,1)
        x = F.relu(self.bn3(self.fc3(x).permute(0,2,1)))
        x = x.permute(0,2,1)
        x = F.relu(self.bn4(self.fc4(x).permute(0,2,1)))
        
        x = x.permute(0,2,1)
       
        return x

class AXform(nn.Module):
    def __init__(self, K1, K2, N):
        super(AXform, self).__init__()
        self.K1 = K1
        self.K2 = K2
        self.N = N  
        self.conv1 = nn.Conv1d(K1, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.softmax = nn.Softmax(dim=2)
        self.conv4 = nn.Conv1d(K2, 3, 1)

    def forward(self, x):
        x_base = x
        x = x.transpose(1, 2).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
       
        x_weights = self.softmax(x)
        
        x = torch.bmm(x_weights, x_base)  
        
        x = x.transpose(1, 2).contiguous()
        x = self.conv4(x)
        x = x.transpose(1, 2).contiguous()

        return x
    
class Decoder_Network(nn.Module):
    def __init__(self, K1, K2, N):
        super(Decoder_Network, self).__init__()
        
        self.num_branch = 8
        self.K1 = K1
        self.K2 = K2
        self.N = N
        
        self.featmap = nn.ModuleList([MappingNet(self.K1,self.N) for i in range(self.num_branch)])
        self.pointgen = nn.ModuleList([AXform(self.K1, self.K2, self.N) for i in range(self.num_branch)])
       
    def forward(self, x, x_part):
       
        x_part = x_part.contiguous() # .contiguous() to make fps work
        x_part = sample_farthest_points(x_part, K=1024)[0]
        x_feat = x
        x_1 = torch.empty(size=(x_part.shape[0], 0, 3)).to(x_part.device)
        
        for i in range(self.num_branch):
            _x_1 = self.pointgen[i](self.featmap[i](x_feat))
            x_1 = torch.cat((x_1, _x_1), dim=1)

        x_coarse = torch.cat((x_1, x_part), dim=1)

        return x_coarse
