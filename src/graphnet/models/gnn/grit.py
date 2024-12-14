import torch
import torch.nn as nn

from graphnet.models.gnn.gnn import GNN

from graphnet.models.components.layers import GritTransformerLayer, SANGraphHead
from graphnet.models.components.embedding import RRWPLinearEdgeEncoder, RRWPLinearNodeEncoder, LinearNodeEncoder, LinearEdgeEncoder

class GRIT(GNN):
    """
    GRIT is a graph transformer model.
    
    Paper: "Graph Inductive Biases in Transformers without Message Passing", https://arxiv.org/abs/2305.17589
    Original code: https://github.com/LiamMa/GRIT
    """
    def __init__(
                 self,
                 nb_inputs,
                 hidden_dim,
                 dim_out=1,
                 ksteps=21,
                 n_layers=10,
                 n_heads=8,
                 pad_to_full_graph=True,
                 add_node_attr_as_self_loop=False,
                 dropout=0.0,
                 fill_value=0.0,
                 layer_norm=False,
                 batch_norm=True,
                 attn_dropout=0.2,
                 edge_enhance=True,
                 update_edges=True,
                 clamp=5.0,
                 act='relu',
                 attn_act='relu',
                 norm_e=True,
                 O_e=True,
                 ):
        super().__init__(nb_inputs, dim_out)
        
        self.node_encoder = LinearNodeEncoder(nb_inputs, hidden_dim)
        self.edge_encoder = LinearEdgeEncoder(hidden_dim)
        
        self.rrwp_abs_encoder = RRWPLinearNodeEncoder(ksteps, hidden_dim)
        self.rrwp_rel_encoder = RRWPLinearEdgeEncoder(ksteps, 
                                                      hidden_dim,
                                                      pad_to_full_graph=pad_to_full_graph,
                                                      add_node_attr_as_self_loop=add_node_attr_as_self_loop,
                                                      fill_value=fill_value)

        layers = []
        for _ in range(n_layers):
            layers.append(GritTransformerLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=n_heads,
                dropout=dropout,
                act=act,
                attn_dropout=attn_dropout,
                layer_norm=layer_norm,
                batch_norm=batch_norm,
                residual=True,
                norm_e=norm_e,
                O_e=O_e,
                update_edges=update_edges,
                attn_act=attn_act,
                attn_clamp=clamp,
                attn_edge_enhance=edge_enhance,
            ))
        self.layers = nn.ModuleList(layers)
        self.head = SANGraphHead(dim_in=hidden_dim, dim_out=1)

    def forward(self, x):
        # Apply linear layers to node/edge features
        x = self.node_encoder(x)
        x = self.edge_encoder(x)
        
        # Encode with RRWP
        x = self.rrwp_abs_encoder(x)
        x = self.rrwp_rel_encoder(x)
        
        # Apply GRIT layers
        for layer in self.layers:
            x = layer(x)

        # Graph head
        x = self.head(x)

        return x