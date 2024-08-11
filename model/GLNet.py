import torch
import torch.nn as nn


class GLNID(nn.Module):
    def __init__(self, num_u, num_i, emb_dim, dev):
        super(GLNID, self).__init__()

        self.device = dev
        self.num_user = num_u
        self.num_item = num_i
        self.emb_dim = emb_dim
        # nodes embedding
        self.user_emb = nn.Embedding(num_u, emb_dim)
        self.item_emb = nn.Embedding(num_i, emb_dim)

        self.buy_edges_emb = nn.Embedding(1, emb_dim)

        a = 0.5

        self.buy_user_user_weights = nn.Parameter(torch.randn(emb_dim)*a)
        self.buy_user_item_weights = nn.Parameter(torch.randn(emb_dim)*a)
        self.buy_item_user_weights = nn.Parameter(torch.randn(emb_dim)*a)
        self.buy_item_item_weights = nn.Parameter(torch.randn(emb_dim)*a)

        self.buy_edges_weights = nn.Parameter(torch.randn(emb_dim, emb_dim)*a)

        self.buy_user_user_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim)*a)
        self.buy_user_item_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim)*a)
        self.buy_item_user_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim)*a)
        self.buy_item_item_weights1 = nn.Parameter(torch.randn(emb_dim, emb_dim)*a)

        # active fun
        self.active = nn.LeakyReLU()

    def forward(self, user_nb, item_nb, edge_nb):
        a = 0.1
        b = 0.1
        x = self.user_emb(user_nb) * a
        y = self.item_emb(item_nb) * a
        edge = self.buy_edges_emb(edge_nb) * b

        x_buy_xx1 = x * edge @ self.buy_user_user_weights
        x_buy_xx2 = x * edge @ self.buy_user_user_weights
        x_buy_xyx = x * edge @ self.buy_user_item_weights
        x_buy_xyy = y * edge @ self.buy_user_item_weights

        # x_buy_xx1 = x @ edge[0]
        # x_buy_xx2 = x @ edge[0]
        # x_buy_xyx = x @ edge[0]
        # x_buy_xyy = y @ edge[0]

        with torch.no_grad():
            ttt = torch.einsum('i,j->ij', x_buy_xx1, x_buy_xx2)
            x_buy_temp1 = ttt @ x
            # x_buy_temp1 = torch.einsum('i,j->ij', x_buy_xx1, x_buy_xx2) @ x
            x_buy_temp2 = torch.einsum('i,j->ij', x_buy_xyx, x_buy_xyy) @ y

        x_buy1 = self.active(x_buy_temp1 @ self.buy_user_user_weights1)
        x_buy2 = self.active(x_buy_temp2 @ self.buy_user_item_weights1)

        #######################################################################

        y_buy_yy1 = y * edge @ self.buy_item_item_weights
        y_buy_yy2 = y * edge @ self.buy_item_item_weights
        y_buy_yxx = y * edge @ self.buy_item_user_weights
        y_buy_yxy = x * edge @ self.buy_item_user_weights

        # y_buy_yy1 = y @ edge[0]
        # y_buy_yy2 = y @ edge[0]
        # y_buy_yxx = y @ edge[0]
        # y_buy_yxy = x @ edge[0]

        with torch.no_grad():
            y_buy_temp1 = torch.einsum('i,j->ij', y_buy_yy1, y_buy_yy2) @ y
            y_buy_temp2 = torch.einsum('i,j->ij', y_buy_yxx, y_buy_yxy) @ x

        y_buy1 = self.active(y_buy_temp1 @ self.buy_item_item_weights1)
        y_buy2 = self.active(y_buy_temp2 @ self.buy_item_user_weights1)

        new = self.active(edge @ self.buy_edges_weights)

        x_buy = (x_buy1 + x_buy2) / 2
        y_buy = (y_buy1 + y_buy2) / 2

        x = (x_buy + x)/2
        y = (y_buy + y)/2
        e = (new+edge)/2

        return x, y, e


class ItemEmb(torch.nn.Module):
    def __init__(self):
        super(ItemEmb, self).__init__()
        self.num_rate = 6
        self.num_genre = 25
        self.num_director = 2186
        self.num_actor = 8030
        self.embedding_dim = 64

        self.embedding_rate = torch.nn.Embedding(
            num_embeddings=self.num_rate,
            embedding_dim=self.embedding_dim
        )

        self.embedding_genre = torch.nn.Linear(
            in_features=self.num_genre,
            out_features=self.embedding_dim,
            bias=False
        )

        self.embedding_director = torch.nn.Linear(
            in_features=self.num_director,
            out_features=self.embedding_dim,
            bias=False
        )

        self.embedding_actor = torch.nn.Linear(
            in_features=self.num_actor,
            out_features=self.embedding_dim,
            bias=False
        )

        self.lin = torch.nn.Linear(self.embedding_dim * 4, self.embedding_dim)

    def forward(self, rate_idx, genre_idx, director_idx, actors_idx, vars=None):
        rate_emb = self.embedding_rate(rate_idx)
        genre_emb = self.embedding_genre(genre_idx.float()) / torch.sum(genre_idx.float(), 1).view(-1, 1)
        director_emb = self.embedding_director(director_idx.float()) / torch.sum(director_idx.float(), 1).view(-1, 1)
        actors_emb = self.embedding_actor(actors_idx.float()) / torch.sum(actors_idx.float(), 1).view(-1, 1)
        return self.lin(torch.cat((rate_emb, genre_emb, director_emb, actors_emb), 1))


class UserEmb(torch.nn.Module):
    def __init__(self):
        super(UserEmb, self).__init__()
        self.num_gender = 98
        self.num_age = 7
        self.num_occupation = 21
        self.num_zipcode = 3402
        self.embedding_dim = 64

        self.embedding_gender = torch.nn.Embedding(
            num_embeddings=self.num_gender,
            embedding_dim=self.embedding_dim
        )

        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim
        )

        self.embedding_occupation = torch.nn.Embedding(
            num_embeddings=self.num_occupation,
            embedding_dim=self.embedding_dim
        )

        self.embedding_area = torch.nn.Embedding(
            num_embeddings=self.num_zipcode,
            embedding_dim=self.embedding_dim
        )
        self.lin = torch.nn.Linear(self.embedding_dim * 4, self.embedding_dim)

    def forward(self, gender_idx, age_idx, occupation_idx, area_idx):
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        area_emb = self.embedding_area(area_idx)
        return self.lin(torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1))


class GLNFeat(nn.Module):
    def __init__(self, emb_dim, dev):
        super(GLNFeat, self).__init__()

        self.device = dev
        self.emb_dim = emb_dim
        # nodes embedding
        self.user_emb = UserEmb().to(dev)
        self.item_emb = ItemEmb().to(dev)
        self.edges_emb = nn.Embedding(1, emb_dim)

        a = 0.5

        # self.node_weights = nn.ParameterDict()
        # self.edge_weights = nn.ParameterDict()
        # self.node_weights1 = nn.ParameterDict()

        self.user_user_weights_rate = nn.Parameter(torch.randn(emb_dim)*a)
        self.user_item_weights_rate = nn.Parameter(torch.randn(emb_dim)*a)
        self.item_user_weights_rate = nn.Parameter(torch.randn(emb_dim)*a)
        self.item_item_weights_rate = nn.Parameter(torch.randn(emb_dim)*a)

        self.edges_weights_rate = nn.Parameter(torch.randn(emb_dim, emb_dim)*a)

        self.user_user_weights1_rate = nn.Parameter(torch.randn(emb_dim, emb_dim)*a)
        self.user_item_weights1_rate = nn.Parameter(torch.randn(emb_dim, emb_dim)*a)
        self.item_user_weights1_rate = nn.Parameter(torch.randn(emb_dim, emb_dim)*a)
        self.item_item_weights1_rate = nn.Parameter(torch.randn(emb_dim, emb_dim)*a)

        # active fun
        self.active = nn.LeakyReLU()

    def forward(self, user_emb, item_emb, edge_emb):
        a = 0.12
        b = 0.12
        # user
        gender_idx = user_emb[:, 0]
        age_idx = user_emb[:, 1]
        occupation_idx = user_emb[:, 2]
        area_idx = user_emb[:, 3]
        # item
        rate_idx = item_emb[:, 0]
        genre_idx = item_emb[:, 1:26]
        director_idx = item_emb[:, 26:2212]
        actor_idx = item_emb[:, 2212:10242]

        x = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx) * a
        y = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx) * a
        edge = self.edges_emb(edge_emb) * b

        x_buy_xx1 = x * edge @ self.user_user_weights_rate
        x_buy_xx2 = x * edge @ self.user_user_weights_rate
        x_buy_xyx = x * edge @ self.user_item_weights_rate
        x_buy_xyy = y * edge @ self.user_item_weights_rate

        with torch.no_grad():
            x_buy_temp1 = torch.einsum('i,j->ij', x_buy_xx1, x_buy_xx2) @ x
            x_buy_temp2 = torch.einsum('i,j->ij', x_buy_xyx, x_buy_xyy) @ y

        x_buy1 = self.active(x_buy_temp1 @ self.user_user_weights1_rate)
        x_buy2 = self.active(x_buy_temp2 @ self.user_item_weights1_rate)

        #######################################################################

        y_buy_yy1 = y * edge @ self.item_item_weights_rate
        y_buy_yy2 = y * edge @ self.item_item_weights_rate
        y_buy_yxx = y * edge @ self.item_user_weights_rate
        y_buy_yxy = x * edge @ self.item_user_weights_rate

        with torch.no_grad():
            y_buy_temp1 = torch.einsum('i,j->ij', y_buy_yy1, y_buy_yy2) @ y
            y_buy_temp2 = torch.einsum('i,j->ij', y_buy_yxx, y_buy_yxy) @ x

        y_buy1 = self.active(y_buy_temp1 @ self.item_item_weights1_rate)
        y_buy2 = self.active(y_buy_temp2 @ self.item_user_weights1_rate)

        new = self.active(edge @ self.edges_weights_rate)

        x_buy = (x_buy1 + x_buy2) / 2
        y_buy = (y_buy1 + y_buy2) / 2

        x = (x_buy + x) / 2
        y = (y_buy + y) / 2
        e = (new + edge) / 2

        return x, y, e
