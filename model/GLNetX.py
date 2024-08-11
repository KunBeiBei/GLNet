# LightGCN-GLN
# EHCF-GLN
# GHCF-GLN
# HPMR-GLN

# 多行为+边
import torch
import torch.nn as nn
# import dgl
# import numpy as np
# import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv


class GHCF(nn.Module):
    def __init__(self, num_user, num_item, emb_dim):
        super(GHCF, self).__init__()
        # 结点嵌入
        self.user_emb = nn.Embedding(num_user, emb_dim)
        self.item_emb = nn.Embedding(num_item, emb_dim)
        # 边类型嵌入
        self.edges_emb = nn.Embedding(3, emb_dim)
        # a.repeat(n, 1)

        # 节点权重
        self.weight_1 = nn.Parameter(torch.rand(emb_dim, emb_dim))
        self.weight_2 = nn.Parameter(torch.rand(emb_dim, emb_dim))
        self.weight_3 = nn.Parameter(torch.rand(emb_dim, emb_dim))
        self.weight_4 = nn.Parameter(torch.rand(emb_dim, emb_dim))
        # 边权重
        self.edge_weight_1 = nn.Parameter(torch.rand(emb_dim, emb_dim))
        self.edge_weight_2 = nn.Parameter(torch.rand(emb_dim, emb_dim))
        self.edge_weight_3 = nn.Parameter(torch.rand(emb_dim, emb_dim))
        self.edge_weight_4 = nn.Parameter(torch.rand(emb_dim, emb_dim))

        # buy
        # gcn
        self.buy_conv1_user = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)
        self.buy_conv1_item = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)

        self.buy_conv2_user = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)
        self.buy_conv2_item = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)

        self.buy_conv3_user = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)
        self.buy_conv3_item = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)

        self.buy_conv4_user = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)
        self.buy_conv4_item = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)

        # cart
        # gcn
        self.cart_conv1_user = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)
        self.cart_conv1_item = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)

        self.cart_conv2_user = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)
        self.cart_conv2_item = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)

        self.cart_conv3_user = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)
        self.cart_conv3_item = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)

        self.cart_conv4_user = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)
        self.cart_conv4_item = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)

        # pv
        # gcn
        self.pv_conv1_user = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)
        self.pv_conv1_item = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)

        self.pv_conv2_user = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)
        self.pv_conv2_item = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)

        self.pv_conv3_user = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)
        self.pv_conv3_item = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)

        self.pv_conv4_user = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)
        self.pv_conv4_item = GraphConv(emb_dim, emb_dim, activation=nn.LeakyReLU(), bias=False, weight=False)

        # 激活函数
        self.active = nn.LeakyReLU()

        self.w_pv = 0 / 6
        self.w_cart = 5 / 6
        self.w_buy = 1 / 6

        self.alpha = 1
        self.beta = 0

        self.decay = 10

    def forward(self, g, e_type):
        src_feature = self.user_emb(g['buy'].nodes('user')) * 0.01
        dst_feature = self.item_emb(g['buy'].nodes('item')) * 0.01
        edge_feature = self.edges_emb(e_type) * 0.01
        edge_buy = edge_feature[0].repeat(g['buy'].num_edges(), 1)
        edge_pv = edge_feature[1].repeat(g['pv'].num_edges(), 1)
        edge_cart = edge_feature[2].repeat(g['cart'].num_edges(), 1)

        # 第一层

        # 处理节点
        buy_dst_feature_1 = self.buy_conv1_item(g['buy'], (src_feature, dst_feature), weight=self.weight_1,
                                                edge_weight=edge_buy)
        buy_src_feature_1 = self.buy_conv1_user(g['rev_buy'], (dst_feature, src_feature), weight=self.weight_1,
                                                edge_weight=edge_buy)

        cart_dst_feature_1 = self.cart_conv1_item(g['cart'], (src_feature, dst_feature), weight=self.weight_1,
                                                  edge_weight=edge_cart)
        cart_src_feature_1 = self.cart_conv1_user(g['rev_cart'], (dst_feature, src_feature), weight=self.weight_1,
                                                  edge_weight=edge_cart)

        pv_dst_feature_1 = self.pv_conv1_item(g['pv'], (src_feature, dst_feature), weight=self.weight_1,
                                              edge_weight=edge_pv)
        pv_src_feature_1 = self.pv_conv1_user(g['rev_pv'], (dst_feature, src_feature), weight=self.weight_1,
                                              edge_weight=edge_pv)

        dst_feature_1 = self.w_buy * buy_dst_feature_1 + self.w_cart * cart_dst_feature_1 + self.w_pv * pv_dst_feature_1
        src_feature_1 = self.w_buy * buy_src_feature_1 + self.w_cart * cart_src_feature_1 + self.w_pv * pv_src_feature_1

        # 处理边
        edge_buy_1 = torch.matmul(edge_buy, self.edge_weight_1)
        edge_cart_1 = torch.matmul(edge_cart, self.edge_weight_1)
        edge_pv_1 = torch.matmul(edge_pv, self.edge_weight_1)

        # 第二层
        # 处理节点
        buy_dst_feature_2 = self.buy_conv2_item(g['buy'], (src_feature_1, dst_feature_1), weight=self.weight_2,
                                                edge_weight=edge_buy_1)
        buy_src_feature_2 = self.buy_conv2_user(g['rev_buy'], (dst_feature_1, src_feature_1), weight=self.weight_2,
                                                edge_weight=edge_buy_1)

        cart_dst_feature_2 = self.cart_conv2_item(g['cart'], (src_feature_1, dst_feature_1), weight=self.weight_2,
                                                  edge_weight=edge_cart_1)
        cart_src_feature_2 = self.cart_conv2_user(g['rev_cart'], (dst_feature_1, src_feature_1), weight=self.weight_2,
                                                  edge_weight=edge_cart_1)

        pv_dst_feature_2 = self.pv_conv2_item(g['pv'], (src_feature_1, dst_feature_1), weight=self.weight_2,
                                              edge_weight=edge_pv_1)
        pv_src_feature_2 = self.pv_conv2_user(g['rev_pv'], (dst_feature_1, src_feature_1), weight=self.weight_2,
                                              edge_weight=edge_pv_1)

        dst_feature_2 = self.w_buy * buy_dst_feature_2 + self.w_cart * cart_dst_feature_2 + self.w_pv * pv_dst_feature_2
        src_feature_2 = self.w_buy * buy_src_feature_2 + self.w_cart * cart_src_feature_2 + self.w_pv * pv_src_feature_2

        # 处理边
        edge_buy_2 = torch.matmul(edge_buy_1, self.edge_weight_2)
        edge_cart_2 = torch.matmul(edge_cart_1, self.edge_weight_2)
        edge_pv_2 = torch.matmul(edge_pv_1, self.edge_weight_2)

        # 第三层
        # 处理节点
        buy_dst_feature_3 = self.buy_conv3_item(g['buy'], (src_feature_2, dst_feature_2), weight=self.weight_3,
                                                edge_weight=edge_buy_2)
        buy_src_feature_3 = self.buy_conv3_user(g['rev_buy'], (dst_feature_2, src_feature_2), weight=self.weight_3,
                                                edge_weight=edge_buy_2)

        cart_dst_feature_3 = self.cart_conv3_item(g['cart'], (src_feature_2, dst_feature_2), weight=self.weight_3,
                                                  edge_weight=edge_cart_2)
        cart_src_feature_3 = self.cart_conv3_user(g['rev_cart'], (dst_feature_2, src_feature_2), weight=self.weight_3,
                                                  edge_weight=edge_cart_2)

        pv_dst_feature_3 = self.pv_conv3_item(g['pv'], (src_feature_2, dst_feature_2), weight=self.weight_3,
                                              edge_weight=edge_pv_2)
        pv_src_feature_3 = self.pv_conv3_user(g['rev_pv'], (dst_feature_2, src_feature_2), weight=self.weight_3,
                                              edge_weight=edge_pv_2)

        dst_feature_3 = self.w_buy * buy_dst_feature_3 + self.w_cart * cart_dst_feature_3 + self.w_pv * pv_dst_feature_3
        src_feature_3 = self.w_buy * buy_src_feature_3 + self.w_cart * cart_src_feature_3 + self.w_pv * pv_src_feature_3

        # 处理边
        edge_buy_3 = torch.matmul(edge_buy_2, self.edge_weight_3)
        edge_cart_3 = torch.matmul(edge_cart_2, self.edge_weight_3)
        edge_pv_3 = torch.matmul(edge_pv_2, self.edge_weight_3)

        # 第四层
        # 处理节点
        buy_dst_feature_4 = self.buy_conv4_item(g['buy'], (src_feature_3, dst_feature_3), weight=self.weight_4,
                                                edge_weight=edge_buy_3)
        buy_src_feature_4 = self.buy_conv4_user(g['rev_buy'], (dst_feature_3, src_feature_3), weight=self.weight_4,
                                                edge_weight=edge_buy_3)

        cart_dst_feature_4 = self.cart_conv4_item(g['cart'], (src_feature_3, dst_feature_3), weight=self.weight_4,
                                                  edge_weight=edge_cart_3)
        cart_src_feature_4 = self.cart_conv4_user(g['rev_cart'], (dst_feature_3, src_feature_3), weight=self.weight_4,
                                                  edge_weight=edge_cart_3)

        pv_dst_feature_4 = self.pv_conv4_item(g['pv'], (src_feature_3, dst_feature_3), weight=self.weight_4,
                                              edge_weight=edge_pv_3)
        pv_src_feature_4 = self.pv_conv4_user(g['rev_pv'], (dst_feature_3, src_feature_3), weight=self.weight_4,
                                              edge_weight=edge_pv_3)

        dst_feature_4 = self.w_buy * buy_dst_feature_4 + self.w_cart * cart_dst_feature_4 + self.w_pv * pv_dst_feature_4
        src_feature_4 = self.w_buy * buy_src_feature_4 + self.w_cart * cart_src_feature_4 + self.w_pv * pv_src_feature_4

        # 处理边
        edge_buy_4 = torch.matmul(edge_buy_3, self.edge_weight_4)
        edge_cart_4 = torch.matmul(edge_cart_3, self.edge_weight_4)
        edge_pv_4 = torch.matmul(edge_pv_3, self.edge_weight_4)

        # return src_feature_4, dst_feature_4, edge_buy_4, edge_cart_4, edge_pv_4

        return (src_feature + src_feature_1 + src_feature_2 + src_feature_3 + src_feature_4) / 5, \
               (dst_feature + dst_feature_1 + dst_feature_2 + dst_feature_3 + dst_feature_4) / 5, \
               (edge_buy + edge_buy_1 + edge_buy_2 + edge_buy_3 + edge_buy_4) / 5, (
                           edge_cart + edge_cart_1 + edge_cart_2 + edge_cart_3 + edge_cart_4) / 5, (
                           edge_pv + edge_pv_1 + edge_pv_2 + edge_pv_3 + edge_pv_4) / 5


class GHCF_GLN(nn.Module):
    def __init__(self,
                 num_user, num_item, emb_dim,  # nodes and edges
                 MTL_w  # MTL weights
                 ):
        super(GHCF_GLN, self).__init__()
        # nodes embeddding
        self.user_emb = nn.Embedding(num_user, emb_dim)
        self.item_emb = nn.Embedding(num_item, emb_dim)

        self.buy_edges_emb = nn.Embedding(1, emb_dim)
        self.cart_edges_emb = nn.Embedding(1, emb_dim)
        self.pv_edges_emb = nn.Embedding(1, emb_dim)

        self.buy_user_user_weights = nn.Parameter(torch.randn(emb_dim) / 10)
        self.buy_user_item_weights = nn.Parameter(torch.randn(emb_dim) / 10)
        self.buy_item_user_weights = nn.Parameter(torch.randn(emb_dim) / 10)
        self.buy_item_item_weights = nn.Parameter(torch.randn(emb_dim) / 10)

        self.cart_user_user_weights = nn.Parameter(torch.randn(emb_dim) / 10)
        self.cart_user_item_weights = nn.Parameter(torch.randn(emb_dim) / 10)
        self.cart_item_user_weights = nn.Parameter(torch.randn(emb_dim) / 10)
        self.cart_item_item_weights = nn.Parameter(torch.randn(emb_dim) / 10)

        self.pv_user_user_weights = nn.Parameter(torch.randn(emb_dim) / 10)
        self.pv_user_item_weights = nn.Parameter(torch.randn(emb_dim) / 10)
        self.pv_item_user_weights = nn.Parameter(torch.randn(emb_dim) / 10)
        self.pv_item_item_weights = nn.Parameter(torch.randn(emb_dim) / 10)

        self.buy_user_user_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)
        self.buy_user_item_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)
        self.buy_item_user_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)
        self.buy_item_item_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)

        self.cart_user_user_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)
        self.cart_user_item_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)
        self.cart_item_user_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)
        self.cart_item_item_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)

        self.pv_user_user_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)
        self.pv_user_item_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)
        self.pv_item_user_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)
        self.pv_item_item_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)

        self.buy_edges_weights = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)
        self.cart_edges_weights = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)
        self.pv_edges_weights = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)

        # active fun
        self.active = nn.LeakyReLU()

        self.w_pv = MTL_w['pv']  # 0/6
        self.w_cart = MTL_w['cart']  # 5/6
        self.w_buy = MTL_w['buy']  # 1/6

        # self.decay = 1e-2

    def forward(self, g):
        # Beibei
        # a = 0.005
        # b = 0.005
        # Taobao
        a = 0.1
        b = 0.1

        src_feature = self.user_emb(g['buy'].nodes('user')) * a
        dst_feature = self.item_emb(g['buy'].nodes('item')) * a

        edge_buy = self.buy_edges_emb(torch.arange(1, device=g.device)) * b
        edge_cart = self.buy_edges_emb(torch.arange(1, device=g.device)) * b
        edge_pv = self.buy_edges_emb(torch.arange(1, device=g.device)) * b

        x = src_feature
        y = dst_feature

        x_buy_xx = x * edge_buy @ self.buy_user_user_weights
        x_buy_xyx = x * edge_buy @ self.buy_user_item_weights
        x_buy_xyy = y * edge_buy @ self.buy_user_item_weights

        x_cart_xx = x * edge_cart @ self.cart_user_user_weights
        x_cart_xyx = x * edge_cart @ self.cart_user_item_weights
        x_cart_xyy = y * edge_cart @ self.cart_user_item_weights

        x_pv_xx = x * edge_pv @ self.pv_user_user_weights
        x_pv_xyx = x * edge_pv @ self.pv_user_item_weights
        x_pv_xyy = y * edge_pv @ self.pv_user_item_weights
        with torch.no_grad():
            x_buy_temp1 = torch.einsum('i,j->ij', x_buy_xx, x_buy_xx) @ x
            x_buy_temp2 = torch.einsum('i,j->ij', x_buy_xyx, x_buy_xyy) @ y

            x_cart_temp1 = torch.einsum('i,j->ij', x_cart_xx, x_cart_xx) @ x
            x_cart_temp2 = torch.einsum('i,j->ij', x_cart_xyx, x_cart_xyy) @ y

            x_pv_temp1 = torch.einsum('i,j->ij', x_pv_xx, x_pv_xx) @ x
            x_pv_temp2 = torch.einsum('i,j->ij', x_pv_xyx, x_pv_xyy) @ y

        x_buy1 = x_buy_temp1 @ self.buy_user_user_weights1
        x_buy2 = x_buy_temp2 @ self.buy_user_item_weights1
        x_cart1 = x_cart_temp1 @ self.cart_user_user_weights1
        x_cart2 = x_cart_temp2 @ self.cart_user_item_weights1
        x_pv1 = x_pv_temp1 @ self.pv_user_user_weights1
        x_pv2 = x_pv_temp2 @ self.pv_user_item_weights1
        #######################################################################
        y_buy_yy = y * edge_buy @ self.buy_item_item_weights
        y_buy_yxx = y * edge_buy @ self.buy_item_user_weights
        y_buy_yxy = x * edge_buy @ self.buy_item_user_weights

        y_cart_yy = y * edge_cart @ self.cart_item_item_weights
        y_cart_yxx = y * edge_cart @ self.cart_item_user_weights
        y_cart_yxy = x * edge_cart @ self.cart_item_user_weights

        y_pv_yy = y * edge_pv @ self.pv_item_item_weights
        y_pv_yxx = y * edge_pv @ self.pv_item_user_weights
        y_pv_yxy = x * edge_pv @ self.pv_item_user_weights
        with torch.no_grad():
            y_buy_temp1 = torch.einsum('i,j->ij', y_buy_yy, y_buy_yy) @ y
            y_buy_temp2 = torch.einsum('i,j->ij', y_buy_yxx, y_buy_yxy) @ x

            y_cart_temp1 = torch.einsum('i,j->ij', y_cart_yy, y_cart_yy) @ y
            y_cart_temp2 = torch.einsum('i,j->ij', y_cart_yxx, y_cart_yxy) @ x

            y_pv_temp1 = torch.einsum('i,j->ij', y_pv_yy, y_pv_yy) @ y
            y_pv_temp2 = torch.einsum('i,j->ij', y_pv_yxx, y_pv_yxy) @ x

        y_buy1 = y_buy_temp1 @ self.buy_item_item_weights1
        y_buy2 = y_buy_temp2 @ self.buy_item_user_weights1
        y_cart1 = y_cart_temp1 @ self.cart_item_item_weights1
        y_cart2 = y_cart_temp2 @ self.cart_item_user_weights1
        y_pv1 = y_pv_temp1 @ self.pv_item_item_weights1
        y_pv2 = y_pv_temp2 @ self.pv_item_user_weights1

        new_buy = edge_buy @ self.buy_edges_weights
        new_cart = edge_cart @ self.cart_edges_weights
        new_pv = edge_pv @ self.pv_edges_weights

        x_buy = (x_buy1 + x_buy2) / 2
        y_buy = (y_buy1 + y_buy2) / 2
        x_cart = (x_cart1 + x_cart2) / 2
        y_cart = (y_cart1 + y_cart2) / 2
        x_pv = (x_pv1 + x_pv2) / 2
        y_pv = (y_pv1 + y_pv2) / 2

        x = self.w_buy * x_buy + self.w_cart * x_cart + self.w_pv * x_pv
        y = self.w_buy * y_buy + self.w_cart * y_cart + self.w_pv * y_pv

        x = (x + src_feature) / 2
        y = (y + dst_feature) / 2

        return x, y, (new_buy + edge_buy) / 2, (new_cart + edge_cart) / 2, (new_pv + edge_pv) / 2


class LightGCN_GLN(nn.Module):
    def __init__(self,
                 num_user, num_item, emb_dim,  # nodes and edges
                 MTL_w  # MTL weights
                 ):
        super(LightGCN_GLN, self).__init__()
        # nodes embeddding
        self.user_emb = nn.Embedding(num_user, emb_dim)
        self.item_emb = nn.Embedding(num_item, emb_dim)

        self.buy_user_user_weights = nn.Parameter(torch.randn(emb_dim) / 10)
        self.buy_user_item_weights = nn.Parameter(torch.randn(emb_dim) / 10)
        self.buy_item_user_weights = nn.Parameter(torch.randn(emb_dim) / 10)
        self.buy_item_item_weights = nn.Parameter(torch.randn(emb_dim) / 10)

        self.cart_user_user_weights = nn.Parameter(torch.randn(emb_dim) / 10)
        self.cart_user_item_weights = nn.Parameter(torch.randn(emb_dim) / 10)
        self.cart_item_user_weights = nn.Parameter(torch.randn(emb_dim) / 10)
        self.cart_item_item_weights = nn.Parameter(torch.randn(emb_dim) / 10)

        self.pv_user_user_weights = nn.Parameter(torch.randn(emb_dim) / 10)
        self.pv_user_item_weights = nn.Parameter(torch.randn(emb_dim) / 10)
        self.pv_item_user_weights = nn.Parameter(torch.randn(emb_dim) / 10)
        self.pv_item_item_weights = nn.Parameter(torch.randn(emb_dim) / 10)

        self.buy_user_user_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)
        self.buy_user_item_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)
        self.buy_item_user_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)
        self.buy_item_item_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)

        self.cart_user_user_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)
        self.cart_user_item_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)
        self.cart_item_user_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)
        self.cart_item_item_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)

        self.pv_user_user_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)
        self.pv_user_item_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)
        self.pv_item_user_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)
        self.pv_item_item_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim) / 10)

        # active fun
        self.active = nn.LeakyReLU()

        self.w_pv = MTL_w['pv']  # 0/6
        self.w_cart = MTL_w['cart']  # 5/6
        self.w_buy = MTL_w['buy']  # 1/6

        # self.decay = 1e-2
        self.buy_edges = nn.Parameter(torch.randn(1, emb_dim) / 10)
        self.cart_edges = nn.Parameter(torch.randn(1, emb_dim) / 10)
        self.pv_edges = nn.Parameter(torch.randn(1, emb_dim) / 10)

    def forward(self, g):
        # Beibei
        # a = 0.005
        # b = 0.005
        # Taobao
        a = 0.1
        b = 0.1

        src_feature = self.user_emb(g['buy'].nodes('user')) * a
        dst_feature = self.item_emb(g['buy'].nodes('item')) * a

        x = src_feature
        y = dst_feature

        x_buy_xx = x @ self.buy_user_user_weights
        x_buy_xyx = x @ self.buy_user_item_weights
        x_buy_xyy = y @ self.buy_user_item_weights

        x_cart_xx = x @ self.cart_user_user_weights
        x_cart_xyx = x @ self.cart_user_item_weights
        x_cart_xyy = y @ self.cart_user_item_weights

        x_pv_xx = x @ self.pv_user_user_weights
        x_pv_xyx = x @ self.pv_user_item_weights
        x_pv_xyy = y @ self.pv_user_item_weights

        with torch.no_grad():
            x_buy_temp1 = torch.einsum('i,j->ij', x_buy_xx, x_buy_xx) @ x
            x_buy_temp2 = torch.einsum('i,j->ij', x_buy_xyx, x_buy_xyy) @ y

            x_cart_temp1 = torch.einsum('i,j->ij', x_cart_xx, x_cart_xx) @ x
            x_cart_temp2 = torch.einsum('i,j->ij', x_cart_xyx, x_cart_xyy) @ y

            x_pv_temp1 = torch.einsum('i,j->ij', x_pv_xx, x_pv_xx) @ x
            x_pv_temp2 = torch.einsum('i,j->ij', x_pv_xyx, x_pv_xyy) @ y

        x_buy1 = x_buy_temp1 @ self.buy_user_user_weights1
        x_buy2 = x_buy_temp2 @ self.buy_user_item_weights1
        x_cart1 = x_cart_temp1 @ self.cart_user_user_weights1
        x_cart2 = x_cart_temp2 @ self.cart_user_item_weights1
        x_pv1 = x_pv_temp1 @ self.pv_user_user_weights1
        x_pv2 = x_pv_temp2 @ self.pv_user_item_weights1

        #######################################################################

        y_buy_yy = y @ self.buy_item_item_weights
        y_buy_yxx = y @ self.buy_item_user_weights
        y_buy_yxy = x @ self.buy_item_user_weights

        y_cart_yy = y @ self.cart_item_item_weights
        y_cart_yxx = y @ self.cart_item_user_weights
        y_cart_yxy = x @ self.cart_item_user_weights

        y_pv_yy = y @ self.pv_item_item_weights
        y_pv_yxx = y @ self.pv_item_user_weights
        y_pv_yxy = x @ self.pv_item_user_weights

        with torch.no_grad():
            y_buy_temp1 = torch.einsum('i,j->ij', y_buy_yy, y_buy_yy) @ y
            y_buy_temp2 = torch.einsum('i,j->ij', y_buy_yxx, y_buy_yxy) @ x

            y_cart_temp1 = torch.einsum('i,j->ij', y_cart_yy, y_cart_yy) @ y
            y_cart_temp2 = torch.einsum('i,j->ij', y_cart_yxx, y_cart_yxy) @ x

            y_pv_temp1 = torch.einsum('i,j->ij', y_pv_yy, y_pv_yy) @ y
            y_pv_temp2 = torch.einsum('i,j->ij', y_pv_yxx, y_pv_yxy) @ x

        y_buy1 = y_buy_temp1 @ self.buy_item_item_weights1
        y_buy2 = y_buy_temp2 @ self.buy_item_user_weights1
        y_cart1 = y_cart_temp1 @ self.cart_item_item_weights1
        y_cart2 = y_cart_temp2 @ self.cart_item_user_weights1
        y_pv1 = y_pv_temp1 @ self.pv_item_item_weights1
        y_pv2 = y_pv_temp2 @ self.pv_item_user_weights1

        x_buy = (x_buy1 + x_buy2) / 2
        y_buy = (y_buy1 + y_buy2) / 2
        x_cart = (x_cart1 + x_cart2) / 2
        y_cart = (y_cart1 + y_cart2) / 2
        x_pv = (x_pv1 + x_pv2) / 2
        y_pv = (y_pv1 + y_pv2) / 2

        x = self.w_buy * x_buy + self.w_cart * x_cart + self.w_pv * x_pv
        y = self.w_buy * y_buy + self.w_cart * y_cart + self.w_pv * y_pv

        x = (x + src_feature) / 2
        y = (y + dst_feature) / 2

        return x, y, self.buy_edges, self.cart_edges, self.pv_edges
