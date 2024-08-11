# 多行为+边
import dgl
import tqdm
import torch
import numpy as np
from model import GHCF_GLN as GLN
from model import GHCF
from model import GLNet
from config import parse_args

from evaluate import Recall, NDCG

args = parse_args()


def test(g, model, R_buy, R_test, k, device):
    metrics = []

    for i in k:
        metrics.append(Recall(i))
    for i in k:
        metrics.append(NDCG(i))

    with torch.no_grad():

        model.eval()
        model = model.to(device)
        g = g.to(device)

        src_feature, dst_feature, edge_buy, _, _ = model(g)

        ys = []
        for src in src_feature:
            # 把一个用户复制物品的数量
            src = src.repeat(dst_feature.shape[0], 1)
            y = torch.einsum('ij,kj->i', src * dst_feature, edge_buy)
            ys.append(y.view(-1))
        ys = torch.stack(ys)

        ys = ys.to(torch.device('cpu'))
        R_buy = R_buy.to(torch.device('cpu'))
        R_test = R_test.to(torch.device('cpu'))

        user_item = ys - R_buy * 100000000

        arr = []
        for metric in metrics:
            metric.start()
            metric(user_item, R_test)
            metric.stop()
            arr.append(round(metric.metric, 4))
            print('test:{}:{}'.format(metric.get_title(), round(metric.metric, 4)), end='\t')
        print()
        return arr


if __name__ == '__main__':
    SEED = 2020
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Data loading...")

    dataset = args.dataset

    # embed size
    d = args.embed_size

    c0 = args.TaoBao_c0
    c1 = args.TaoBao_i
    MTL_w = eval(args.MTL_TaoBao_w)
    decay = args.TaoBao_decay
    if dataset == 'Beibei':
        c0 = args.BeiBei_c0
        c1 = args.BeiBei_i
        MTL_w = eval(args.MTL_BeiBei_w)
        decay = args.BeiBei_decay

    traNum1 = 0
    traNum2 = args.epoch
    # directory = 'result/GHCF'
    k = eval(args.Ks)
    l = args.layer_size
    lr = args.lr

    R_buy = torch.load('data/' + dataset + '/R_buy.pth')
    R_test = torch.load('data/' + dataset + '/R_test.pth')
    # R_pv = torch.load(dataset+'/a/R_pv.pth')
    # R_cart = torch.load(dataset+'/a/R_cart.pth')

    R_buy = torch.tensor(R_buy.toarray())
    R_test = torch.tensor(R_test.toarray())

    r_buy_user = torch.load('data/' + dataset + '/r_buy_user.pth')
    r_buy_item = torch.load('data/' + dataset + '/r_buy_item.pth')
    r_pv_user = torch.load('data/' + dataset + '/r_pv_user.pth')
    r_pv_item = torch.load('data/' + dataset + '/r_pv_item.pth')
    r_cart_user = torch.load('data/' + dataset + '/r_cart_user.pth')
    r_cart_item = torch.load('data/' + dataset + '/r_cart_item.pth')

    g = dgl.heterograph({
        ('user', 'buy', 'item'): (r_buy_user, r_buy_item),
        ('user', 'cart', 'item'): (r_cart_user, r_cart_item),
        ('user', 'pv', 'item'): (r_pv_user, r_pv_item),
        ('item', 'rev_buy', 'user'): (r_buy_item, r_buy_user),
        ('item', 'rev_cart', 'user'): (r_cart_item, r_cart_user),
        ('item', 'rev_pv', 'user'): (r_pv_item, r_pv_user)
    })

    Cv_f_buy = 0.01
    Cv_f_cart = 0.01
    Cv_f_pv = 0.01
    if dataset == 'Beibei':
        Cv_f_buy = 0.1
        Cv_f_cart = 0.1
        Cv_f_pv = 0.1

    print("data Loading completed")

    model = GLN(g['buy'].num_nodes('user'), g['buy'].num_nodes('item'), d, MTL_w)
    # model = GHCF(g['buy'].num_nodes('user'), g['buy'].num_nodes('item'), d)
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    for param in model.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    print(f'Total params: {Total_params / 1e6}M')
    print(f'Trainable params: {Trainable_params / 1e6}M')
    print(f'Non-trainable params: {NonTrainable_params / 1e6}M')
    # 1.92128M
    # 7.51682M

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    recall10 = []
    recall50 = []
    recall100 = []
    ndcg10 = []
    ndcg50 = []
    ndcg100 = []
    losss = []

    # train & test
    for epoch in tqdm.tqdm(range(traNum1, traNum2)):
        model.train()

        g = g.to(device)

        # Cv_f_buy = Cv_f_buy.to(device)
        # Cv_f_cart = Cv_f_cart.to(device)
        # Cv_f_pv = Cv_f_pv.to(device)

        model = model.to(device)
        optimizer.zero_grad()

        src_feature, dst_feature, edge_buy, edge_cart, edge_pv = model(g)

        temp = torch.einsum('ab,ac->bc', dst_feature, dst_feature) \
               * torch.einsum('ab,ac->bc', src_feature, src_feature)

        buy1 = src_feature[g['buy'].edges()[0]]
        buy2 = dst_feature[g['buy'].edges()[1]]
        cart1 = src_feature[g['cart'].edges()[0]]
        cart2 = dst_feature[g['cart'].edges()[1]]
        pv1 = src_feature[g['pv'].edges()[0]]
        pv2 = dst_feature[g['pv'].edges()[1]]

        buy_y = torch.einsum('ij,kj->i', buy1 * buy2, edge_buy)
        cart_y = torch.einsum('ij,kj->i', cart1 * cart2, edge_cart)
        pv_y = torch.einsum('ij,kj->i', pv1 * pv2, edge_pv)

        buy_temp = temp * torch.einsum('ab,ac->bc', edge_buy, edge_buy)
        cart_temp = temp * torch.einsum('ab,ac->bc', edge_cart, edge_cart)
        pv_temp = temp * torch.einsum('ab,ac->bc', edge_pv, edge_pv)

        loss1 = Cv_f_buy * torch.sum(buy_temp) \
                + torch.sum((1 - Cv_f_buy) * torch.pow(buy_y, 2) - 2 * buy_y)

        loss2 = Cv_f_cart * torch.sum(cart_temp) \
                + torch.sum((1 - Cv_f_cart) * torch.pow(cart_y, 2) - 2 * cart_y)

        loss3 = Cv_f_pv * torch.sum(pv_temp) \
                + torch.sum((1 - Cv_f_pv) * torch.pow(pv_y, 2) - 2 * pv_y)

        regularizer = torch.norm(src_feature) ** 2 / 2 + torch.norm(dst_feature) ** 2 / 2
        l2_loss = regularizer * decay

        loss = loss1 * model.w_buy + loss2 * model.w_cart + loss3 * model.w_pv + l2_loss

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        losss.append(float(loss))

        if (epoch + 1) % 10 == 0:
            print(float(loss), float(regularizer))
            arr = test(g, model, R_buy, R_test, k, torch.device('cuda'))
            recall10.append(arr[0])
            recall50.append(arr[1])
            recall100.append(arr[2])
            ndcg10.append(arr[3])
            ndcg50.append(arr[4])
            ndcg100.append(arr[5])

    torch.save(recall10, 'jeg/T/new4/recall10.pth')
    torch.save(recall50, 'jeg/T/new4/recall50.pth')
    torch.save(recall100, 'jeg/T/new4/recall100.pth')
    torch.save(ndcg10, 'jeg/T/new4/ndcg10.pth')
    torch.save(ndcg50, 'jeg/T/new4/ndcg50.pth')
    torch.save(ndcg100, 'jeg/T/new4/ndcg100.pth')
    torch.save(losss, 'jeg/T/new4/loss.pth')
