"""
The n3net search function

"""

# -- python --
import torch as th
import torch.nn as nn
from torch.nn.functional import unfold,pad
from einops import rearrange

# -- nat --
from ..nattencuda import NATTENQKRPBFunction

def get_topk(l2_vals,l2_inds,k):

    # -- reshape exh --
    b,nq = l2_vals.shape[:2]
    l2_vals = l2_vals.view(b,nq,-1)
    l2_inds = l2_inds.view(b,nq,-1)
    # l2_inds = l2_inds.view(b,nq,-1,3)

    # -- init --
    vals = th.zeros((b,nq,k),dtype=l2_vals.dtype)
    inds = th.zeros((b,nq,k),dtype=l2_inds.dtype)
    # inds = th.zeros((b,nq,k,3),dtype=l2_inds.dtype)

    # -- take mins --
    order = th.argsort(l2_vals,dim=2,descending=False)
    vals[:,:nq,:] = th.gather(l2_vals,2,order[:,:,:k])
    inds[:,:nq,:] = th.gather(l2_inds,2,order[:,:,:k])
    # for i in range(inds.shape[-1]):
    #     inds[:nq,:,i] = th.gather(l2_inds[:,:,i],1,order[:,:k])

    return vals,inds

class NLSearch():

    def __init__(self,k=7, ps=7, ws=8, nheads=1, chnls=-1, dilation=1,
                 stride0=1, stride1=1, index_reset=True, include_self=False,
                 use_k=True):
        self.k = k
        self.ps = ps
        self.nheads = nheads
        self.ws = ws
        self.chnls = chnls
        self.dilation = dilation
        self.stride0 = stride0
        self.stride1 = stride1
        self.index_reset = index_reset
        self.include_self = include_self
        self.padding = (ps-1)//2
        self.use_k = use_k
        self.index_reset = index_reset
        pdim = (2 * self.ps - 1)
        self.rpb = nn.Parameter(th.zeros(self.nheads, pdim, pdim))

    def __call__(self,vid,foo=0,bar=0):
        if vid.ndim == 5:
            return self.search_batch(vid)
        else:
            return self.search(vid)

    def search_batch(self,vid):
        dists,inds = [],[]
        B = vid.shape[0]
        for b in range(B):
            _dists,_inds = self.search(vid[b])
            dists.append(_dists)
            inds.append(_inds)
        dists = th.stack(dists)
        inds = th.stack(inds)
        return dists,inds

    def search(self,vid):

        # -- patchify --
        xe = vid
        ps = self.ps
        padding = None
        self.rpb = self.rpb.to(vid.device)

        # -- init unfold --
        # x_patch = unfold(vid,(ps,ps),stride=self.stride0,
        #                  dilation=self.dilation)
        # print("x_patch.shape: ",x_patch.shape)
        # x_patch = rearrange(x_patch,'b (h w c) q -> b q h w c',h=ps,w=ps)
        # print("x_patch.shape: ",x_patch.shape)
        x_patch = rearrange(vid,'t (H c) h w -> t H h w c',H=self.nheads)
        print("x_patch.shape: ",x_patch.shape)

        # -- self attn --
        xe_patch = x_patch
        ye_patch = xe_patch

        # -- neighborhood search --
        dists = NATTENQKRPBFunction.apply(xe_patch, ye_patch, self.rpb)
        inds = th.zeros_like(dists).type(th.int32)

        # -- topk --
        if self.use_k:
            dists,inds = get_topk(dists,inds,self.k)

        return dists,inds




