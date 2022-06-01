import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
from spikingjelly.clock_driven import neuron, functional, surrogate, layer

# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, T=100,
                 normalize_input=True, vladv2=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim        
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Sequential(
            nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2),
            nn.BatchNorm2d(num_clusters),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        )
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

        # snn parameter & layer
        self.T = T
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.5),
            nn.Linear(512*30*40, 256, bias=False),
            #neuron.IFNode(surrogate_function=surrogate.ATan()),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.Dropout(0.5),
            nn.Linear(256, 64, bias=False),
            #nn.Linear(512*30*40, 64, bias=False),
            #neuron.IFNode(surrogate_function=surrogate.ATan()),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        )

    def init_params(self, clsts, traindescs):
        #TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :] # sort, descending

            self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            #TODO faiss?
            # knn = NearestNeighbors(n_jobs=-1)
            knn = NearestNeighbors(n_jobs=1)   
            # 否则报错 UserWarning: 
            # Loky-backed parallel loops cannot be called in a multiprocessing, setting n_jobs=1
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:,1] - dsSq[:,0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x):#, write=False):
        N, C = x.shape[:2] #(24,512,30,40)

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim
    
        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        x_flatten = x.view(N, C, -1)

        # print('shape of input feature vectors: ', x_flatten.unsqueeze(0).permute(1, 0, 2, 3).shape)
        # print('shape of centroid: ', self.centroids[0:1,:].shape)
        # print('x_flatten.size(-1)=',x_flatten.size(-1))
        # print('shape of centroid: ', self.centroids[0:1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0).shape)
        # print('shape of soft_assign: ', soft_assign.shape) # (24,64,30*40)

        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            weight = soft_assign[:,C:C+1,:].unsqueeze(2) # *=(24,1,1,1200)
            # with open('./debug_vlad.txt', 'a') as f:
            #     np.set_printoptions(threshold=np.inf)
            #     f.write('===> cluster '+str(C)+' wegiht:\n'+str(soft_assign[:,C])+'\n\n')
            #     #f.write('===> cluster '+str(C)+' residual:\n'+str(residual)+'\n\n')
            #     f.write('===> cluster '+str(C)+' residual with weight:\n'+str((residual*weight))+'\n\n')

            residual *= weight
            vlad[:,C:C+1,:] = residual.sum(dim=-1)

        # # FC-SNN: x_i belong to which cluster (output layer neuron numbers: self.num_clusters)
        # x_flatten = x.view(N, C, -1) #(24,512,1200)        
        # out_spikes_counter = self.fc(x_flatten)
        # for t in range(1, self.T):
        #     out_spikes_counter += self.fc(x_flatten)
        # a = out_spikes_counter / self.T #(24,512,30*40) / (24,64,1200) / (24,64)
        # #print('shape of a: ', a.shape)
        # # _, label = torch.max(a, 1)
        # #print('shape of label: ', label.shape) #(24)
        # #print(label)
        # # if write: 
        # #     with open('./debug.txt', 'a') as f:            
        # #         f.write('weight: '+str(a)+'\n')
        # #         f.write('label: '+str(label)+'\n')
        # #         f.write('\n')

        # # calculate residuals to each clusters
        # vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device) #24*64*512
        # # print('shape of centroid: ', self.centroids[0:1,:].shape)   #(1, 512)        
        # # print('shape of centroid: ', self.centroids[0:1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0).shape) #(1,1,512,1200) 
        # for C in range(self.num_clusters): 
        #     residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
        #                 self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0) #(24,1,512,30*40)   
        #     #residual *= a[:,C:C+1,:].unsqueeze(2) #a.shape=(24,1,1,1200) res.shape=(24,1,512,30*40) 
        #     label_new = (a[:,C:C+1]).repeat(1, residual.size(-1)).unsqueeze(1).unsqueeze(1)

        #     valid={'label':[], 'value':[]}
        #     for i, a_i in enumerate(list(a[:,C].cpu().numpy())):
        #         if a_i:                    
        #             valid['label'].append(i)
        #             valid['value'].append(a_i)
        #     with open('./debug_snn_lif.txt', 'a') as f:
        #         np.set_printoptions(threshold=np.inf)
        #         f.write('===> cluster '+str(C)+' wegiht:\n'+str(a[:,C:C+1])+'\n\n')                    
        #         f.write('===> cluster '+str(C)+' valid:\n'+str(valid)+'\n\n')
        #         f.write('===> cluster '+str(C)+' residual:\n'+str(residual)+'\n\n')
        #         f.write('===> cluster '+str(C)+' residual with weight:\n'+str(residual*label_new)+'\n\n')
            
        #     residual *= label_new                                
        #     vlad[:,C:C+1,:] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad
