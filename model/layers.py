import math
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init
from model.aug import sim_global

class SpatialHeteroModel(nn.Module):
    '''Spatial heterogeneity modeling by using a soft-clustering paradigm.
    '''
    def __init__(self, c_in, nmb_prototype, batch_size, tau=0.5):
        super(SpatialHeteroModel, self).__init__()
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.prototypes = nn.Linear(c_in, nmb_prototype, bias=False)
        
        self.tau = tau
        self.d_model = c_in
        self.batch_size = batch_size

        for m in self.modules():
            self.weights_init(m)
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, z1, z2):
        """Compute the contrastive loss of batched data.
        :param z1, z2 (tensor): shape nlvc
        :param loss: contrastive loss
        """
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = self.l2norm(w)
            self.prototypes.weight.copy_(w)

        zc1 = self.prototypes(self.l2norm(z1.reshape(-1, self.d_model)))

        zc2 = self.prototypes(self.l2norm(z2.reshape(-1, self.d_model)))
        with torch.no_grad():
            q1 = sinkhorn(zc1.detach())
            q2 = sinkhorn(zc2.detach())
        l1 = - torch.mean(torch.sum(q1 * F.log_softmax(zc2 / self.tau, dim=1), dim=1))
        l2 = - torch.mean(torch.sum(q2 * F.log_softmax(zc1 / self.tau, dim=1), dim=1))
        return l1 + l2
    
@torch.no_grad()
def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):
    Q = torch.exp(out / epsilon).t()
    B = Q.shape[1]
    K = Q.shape[0]

    sum_Q = torch.sum(Q)
    Q /= sum_Q
    
    for it in range(sinkhorn_iterations):
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B
    return Q.t()

class TemporalHeteroModel(nn.Module):
    def __init__(self, c_in, batch_size, num_nodes, device):
        super(TemporalHeteroModel, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(num_nodes, c_in))
        self.W2 = nn.Parameter(torch.FloatTensor(num_nodes, c_in)) 
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        
        self.read = AvgReadout()
        self.disc = Discriminator(c_in)
        self.b_xent = nn.BCEWithLogitsLoss()

        lbl_rl = torch.ones(batch_size, num_nodes)
        lbl_fk = torch.zeros(batch_size, num_nodes)
        lbl = torch.cat((lbl_rl, lbl_fk), dim=1)
        if device == 'cuda':
            self.lbl = lbl.cuda()
        
        self.n = batch_size

    def forward(self, z1, z2):
        h = (z1 * self.W1 + z2 * self.W2).squeeze(1) # nlvc->nvc
        s = self.read(h)

        idx = torch.randperm(self.n)
        shuf_h = h[idx]

        logits = self.disc(s, h, shuf_h)
        loss = self.b_xent(logits, self.lbl)
        return loss

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()
        self.sigm = nn.Sigmoid()

    def forward(self, h):
        s = torch.mean(h, dim=1)
        s = self.sigm(s) 
        return s

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.net = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, summary, h_rl, h_fk):
        s = torch.unsqueeze(summary, dim=1)
        s = s.expand_as(h_rl).contiguous()

        sc_rl = torch.squeeze(self.net(h_rl, s), dim=2) 
        sc_fk = torch.squeeze(self.net(h_fk, s), dim=2)

        logits = torch.cat((sc_rl, sc_fk), dim=1)

        return logits

class STEncoder(nn.Module):
    def __init__(self, Kt, Ks, blocks, input_length, num_nodes, droprate=0.1):
        super(STEncoder, self).__init__()
        self.Ks=Ks
        c = blocks[0]
        self.Tconv11 = T_Transformer(64, 8,  4,dropout=False)
        self.pooler = Pooler(input_length - (Kt - 1), c[1])

        self.Sconv12 = S_Transformer(64, 8, 4, dropout=False)
        self.Tconv13 = T_Transformer(64, 8, 4, dropout=False)
        self.ln1 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout1 = nn.Dropout(droprate)

        c = blocks[1]
        self.Tconv21 = T_Transformer(64, 8, 4, dropout=False)
        self.Sconv22 = S_Transformer(64, 8, 4, dropout=False)
        self.Tconv23 = T_Transformer(64, 8, 4, dropout=False)
        self.ln2 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout2 = nn.Dropout(droprate)
        
        self.s_sim_mx = None
        self.t_sim_mx = None
        
        out_len = input_length - 2 * (Kt - 1) * len(blocks)
        self.out_conv = TemporalConvLayer(out_len, c[2], c[2], "GLU")
        self.ln3 = nn.LayerNorm([num_nodes, c[2]])
        self.dropout3 = nn.Dropout(droprate)
        self.receptive_field = input_length + Kt -1

        self.conv11 = nn.Conv2d(1, 64, 1)
        self.conv12 = nn.Conv2d(64, 32, (3, 1))
        self.conv14 = nn.Conv2d(64, 64, (3, 1))

        self.conv22 = nn.Conv2d(64, 64, (3, 1))
        self.conv24 = nn.Conv2d(64, 64, (3, 1))
        self.conv15 = nn.Conv2d(32, 64, 1)





    def forward(self, x0, graph):
        lap_mx = self._cal_laplacian(graph)
        Lk = self._cheb_polynomial(lap_mx, self.Ks)
        
        in_len = x0.size(1)
        if in_len < self.receptive_field:
            x = F.pad(x0, (0,0,0,0,self.receptive_field-in_len,0))
        else:
            x = x0
        x = x.permute(0, 3, 1, 2)

        x = self.conv11(x).permute(0, 3, 2, 1)
        x = self.Tconv11(x)  # (16,170,14,64)
        x = self.conv12(x.permute(0, 3, 2, 1))
        x, x_agg, self.t_sim_mx = self.pooler(x)
        self.s_sim_mx = sim_global(x_agg, sim_type='cos')

        x = self.conv15(x).permute(0, 3, 2, 1)
        x = self.Sconv12(x, Lk)

        x = self.Tconv13(x)
        x = self.conv14(x.permute(0, 3, 2, 1))

        x = self.dropout1(self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

        x = x.permute(0, 3, 2, 1)
        x = self.Tconv21(x)
        x = self.conv22(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        x = self.Sconv22(x, Lk)

        x = self.Tconv23(x)
        x = self.conv24(x.permute(0, 3, 2, 1))

        x = self.dropout2(self.ln2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

        x = self.out_conv(x)
        x = self.dropout3(self.ln3(x.permute(0, 2, 3, 1)))
        return x

    def _cheb_polynomial(self, laplacian, K):
        N = laplacian.size(0)  
        multi_order_laplacian = torch.zeros([K, N, N], device=laplacian.device, dtype=torch.float) 
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k-1]) - \
                                               multi_order_laplacian[k-2]

        return multi_order_laplacian

    def _cal_laplacian(self, graph):
        """
        return the laplacian of the graph.
        :param graph: the graph structure **without** self loop, [v, v].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        I = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype)
        graph = graph + I
        D = torch.diag(torch.sum(graph, dim=-1) ** (-0.5))
        L = I - torch.mm(torch.mm(D, graph), D)
        return L

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x  

class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)  
        return torch.relu(self.conv(x) + x_in)

class S_Transformer(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(S_Transformer, self).__init__()
        self.attention = S_SelfAttention(embed_size, heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.Spatial_embedding = nn.Embedding(170, embed_size)
        self.gcn = SpatioConvLayer(ks=3, c_in=64, c_out=64)
        self.fs = nn.Linear(64, 64)
        self.fg = nn.Linear(64, 64)
    def forward(self, x,graph):
        '''Fixed graph convolution'''
        x0 = x.permute(0, 3, 2, 1)
        X_G = self.gcn(x0, graph).permute(0, 3, 2, 1)

        # Spatial Transformer
        # v = x
        # k = x
        q = x
        B, N, T, C = q.shape
        D_S = self.Spatial_embedding(torch.arange(0, N).to('cuda:0'))
        D_S = D_S.expand(B, T, N, C)  # [B, T, N, C]
        D_S = D_S.permute(0, 2, 1, 3)  # [B, N, T, C]
        query = q + D_S
        attention = self.attention(query, query, query)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        U_S = self.dropout(self.norm2(forward + x))

        g = torch.sigmoid(self.fs(U_S) + self.fg(X_G))
        out = g*U_S + (1-g)*X_G
        return out

class T_Transformer(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(T_Transformer, self).__init__()
        self.attention = T_SelfAttention(embed_size, heads)
        self.one_hot_encoder_time = one_hot_encoder_time(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.temporal_embedding = nn.Embedding(288, embed_size)

    def forward(self, x):
        # x_in = self.align(x)[:, :, self.kt - 1:, :]
        q = x
        B, N, T, C = q.shape  # 32，32，21，128
        D_L = self.temporal_embedding(torch.arange(0, T).to('cuda:0'))
        D_L = D_L.expand(B, N, T, C)

        q = q + D_L

        attention = self.attention(q, q, q)
        x = self.dropout(self.norm1(attention + q))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class SpatioConvLayer(nn.Module):
    def __init__(self, ks, c_in, c_out):
        super(SpatioConvLayer, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1))
        # self.align = Align(c_in, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, Lk):
        x_c = torch.einsum("knm,bitm->bitkn", Lk, x)  
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b 
        # x_in = self.align(x)
        # return torch.relu(x_gc + x_in)
        return torch.relu(x_gc + x)

class Pooler(nn.Module):
    def __init__(self, n_query, d_model, agg='avg'):
        """
        :param n_query: number of query
        :param d_model: dimension of model 
        """
        super(Pooler, self).__init__()

        self.att = FCLayer(d_model, n_query) 
        self.align = Align(d_model, d_model)
        self.softmax = nn.Softmax(dim=2)

        self.d_model = d_model
        self.n_query = n_query 
        if agg == 'avg':
            self.agg = nn.AvgPool2d(kernel_size=(n_query, 1), stride=1)
        elif agg == 'max':
            self.agg = nn.MaxPool2d(kernel_size=(n_query, 1), stride=1)
        else:
            raise ValueError('Pooler supports [avg, max]')
        
    def forward(self, x):
        x_in = self.align(x)[:, :, -self.n_query:, :]
        A = self.att(x)
        A = F.softmax(A, dim=2)

        x = torch.einsum('nclv,nqlv->ncqv', x, A)
        x_agg = self.agg(x).squeeze(2)
        x_agg = torch.einsum('ncv->nvc', x_agg)

        A = torch.einsum('nqlv->lnqv', A)
        A = self.softmax(self.agg(A).squeeze(2))
        return torch.relu(x + x_in), x_agg.detach(), A.detach()


class MLP(nn.Module):
    def __init__(self, c_in, c_out): 
        super(MLP, self).__init__()
        self.fc1 = FCLayer(c_in, int(c_in // 2))
        self.fc2 = FCLayer(int(c_in // 2), c_out)

    def forward(self, x):
        x = torch.tanh(self.fc1(x.permute(0, 3, 1, 2)))
        x = self.fc2(x).permute(0, 2, 3, 1)
        return x

class FCLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(FCLayer, self).__init__()
        self.linear = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        return self.linear(x)



class S_SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(S_SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim * heads == embed_size)

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries):
        B, N, T, C = values.shape

        values = values.reshape(B, N, T, self.heads, self.head_dim)
        keys = keys.reshape(B, N, T, self.heads, self.head_dim)
        queries = queries.reshape(B, N, T, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("bnqhd,bnkhd->bnhqk", [queries, keys])

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=4)

        out = torch.einsum("bnhqk,bnvhd->bnqhd", [attention, values]).reshape(B, N, T, self.heads * self.head_dim)

        out = self.fc_out(out)
        return out

class T_SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(T_SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim * heads == embed_size)

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False).cuda()
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False).cuda()
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False).cuda()
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size).cuda()

    def forward(self, values, keys, queries):
        B, N, T, C = values.shape

        values = values.reshape(B, N, T, self.heads, self.head_dim)
        keys = keys.reshape(B, N, T, self.heads, self.head_dim)
        queries = queries.reshape(B, N, T, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("bnqhd,bnkhd->bnhqk", [queries, keys])

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=4)

        out = torch.einsum("bnhqk,bnvhd->bnqhd", [attention, values]).reshape(B, N, T, self.heads * self.head_dim)

        out = self.fc_out(out)
        return out

class one_hot_encoder_time(nn.Module):
    def __init__(self, embed_size, time_history=12, time_num=288):
        super(one_hot_encoder_time, self).__init__()
        self.time_num = time_num
        # self.day_num = day_num
        self.time_history = time_history
        self.time_I = nn.Parameter(torch.eye(time_num, time_num), requires_grad=True)
        # self.day_I = nn.Parameter(torch.eye(day_num,day_num),requires_grad=True)
        self.onhot_Linear = nn.Linear(time_num, embed_size)

    def forward(self, index, batch_size, node_num):

        if index % self.time_num + self.time_history > self.time_num:
            o1 = self.time_I[index % self.time_num:, :]
            o2 = self.time_I[:(index + self.time_history) % self.time_num, :]
            onehot = torch.cat((o1, o2), 0)
        else:
            onehot = self.time_I[index % self.time_num:index % self.time_num + self.time_history, :]

        onehot = onehot.expand(batch_size, node_num, self.time_history, self.time_num)
        onehot = self.onhot_Linear(onehot)
        return onehot

