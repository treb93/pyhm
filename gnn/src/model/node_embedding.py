
import torch
import torch.nn as nn




class NodeEmbedding(nn.Module):
    """
    Projects the node features into embedding space.
    """

    def __init__(self,
                 in_feats: int,
                 out_feats: int,
                 ):
        
        super().__init__()
        self.proj_feats = nn.Linear(in_feats, out_feats)

    def forward(self,
                node_feats: torch.tensor
            )-> torch.tensor:
        x = self.proj_feats(node_feats.to(torch.float32))
        return x
