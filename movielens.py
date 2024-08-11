# 单行为+边
import tqdm
import torch
import numpy as np
import math
import random
from model import GLNFeat as GLN
from evaluate import get_ndcg
torch.autograd.set_detect_anomaly(True)


def test(test_model,
         user_feat, item_feat, edge_number,
         test_user_id, test_item_id, test_rate, test_user_index,
         test_device):

    with torch.no_grad():

        test_model.eval()
        test_model = test_model.to(test_device)
        user_feat = user_feat.to(test_device)
        item_feat = item_feat.to(test_device)
        edge_number = edge_number.to(test_device)
        test_user_id = test_user_id.to(device)
        test_item_id = test_item_id.to(device)
        test_rate = test_rate.to(device)
        test_user_index = test_user_index.to(device)

        src_test, dst_test, edge = test_model(user_feat, item_feat, edge_number)

        ndcg3 = []
        ndcg5 = []
        for u_id in test_user_index:
            # item样本下标
            dst = dst_test[test_item_id[test_user_id == u_id]]
            # user特征
            src = src_test[u_id]
            src = src.repeat(dst.shape[0], 1)
            # 真实值
            zy = test_rate[test_user_id == u_id]
            # pre
            sd = src * dst
            y = torch.einsum('ij,kj->i', sd, edge)

            _, recommendation_list = y.view(-1).sort(descending = True)
            recommendation_list = recommendation_list.tolist()
            n3 = get_ndcg(recommendation_list, zy.tolist(), 3)
            n5 = get_ndcg(recommendation_list, zy.tolist(), 5)
            ndcg3.append(n3)
            ndcg5.append(n5)

        n3 = np.mean(ndcg3)
        n5 = np.mean(ndcg5)

        return n3, n5


def get_dataset(u_index, u_id, i_id, r):
    t_user_id = []
    t_item_id = []
    t_rate = []
    for i in u_index:
        t_user_id.extend(u_id[u_id == i].tolist())
        t_item_id.extend(i_id[u_id == i].tolist())
        t_rate.extend(r[u_id == i].tolist())
    return torch.tensor(t_user_id), torch.tensor(t_item_id), torch.tensor(t_rate)


if __name__ == '__main__':
    SEED = 2020
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Data loading...")

    # dataset = args.dataset
    dataset = 'movielens_1m'

    # embed size
    d = 64

    traNum1 = 0
    traNum2 = 25000
    lr = 1e-4 # args.lr
    decay = 1e-2

    user_feat = torch.load('data/' + dataset + '/user_feat.pth')
    item_feat = torch.load('data/' + dataset + '/item_feat.pth')
    user_id = torch.load('data/' + dataset + '/user_id.pth') - 1
    item_id = torch.load('data/' + dataset + '/item_id.pth') - 1
    rate = torch.load('data/' + dataset + '/rate.pth')

    user_id = user_id[rate != 3]
    item_id = item_id[rate != 3]
    rate = rate[rate != 3]

    indices = list(range(6040))
    random.shuffle(indices)
    indices = torch.tensor(indices)
    train_user_index = indices[1208:]
    test_user_index = indices[0:1208]

    train_user_id, train_item_id, train_rate = get_dataset(train_user_index, user_id, item_id, rate)
    test_user_id, test_item_id, test_rate = get_dataset(test_user_index, user_id, item_id, rate)

    Cv_f = 0.01

    print("data Loading completed")

    # num_user = 6040 * 0.8
    # num_item = 3704

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model = GLN(d, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mses = []
    maes = []

    # train & test
    for epoch in tqdm.tqdm(range(traNum1, 25000)):
        model.train()

        # Cv_f_buy = Cv_f_buy.to(device)

        model = model.to(device)
        optimizer.zero_grad()

        edge_number = torch.arange(1).to(device)

        item_feat = item_feat.to(device)

        train_user_feat = user_feat[train_user_index].to(device)

        temp_feature = torch.zeros(user_feat.shape[0], 64).to(device)

        src_feature, dst_feature, eee = model(train_user_feat, item_feat, edge_number)

        temp = torch.einsum('ab,ac->bc', dst_feature, dst_feature) * torch.einsum('ab,ac->bc', src_feature, src_feature)

        temp_feature[train_user_index] = src_feature

        rate1 = torch.cat([temp_feature[train_user_id[train_rate == 4]], temp_feature[train_user_id[train_rate == 5]]], 0)
        rate2 = torch.cat([dst_feature[train_item_id[train_rate == 4]], dst_feature[train_item_id[train_rate == 5]]], 0)

        rate_y = torch.einsum('ij,kj->i', rate1 * rate2, eee)

        buy_temp = temp * torch.einsum('ab,ac->bc', eee, eee)

        loss1 = Cv_f * torch.sum(buy_temp) + torch.sum((1 - Cv_f) * torch.pow(rate_y, 2) - 2 * rate_y)

        reg = torch.norm(src_feature) ** 2 / 2 + torch.norm(dst_feature) ** 2 / 2

        l2_loss = reg * decay

        loss = loss1 + l2_loss

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        if (epoch+1) % 100 == 0:
            mse, mae = test(model,
                            user_feat, item_feat, edge_number,
                            test_user_id, test_item_id, test_rate, test_user_index,
                            torch.device('cuda'))

            buy1_0 = torch.cat([temp_feature[train_user_id[train_rate == 1]], temp_feature[train_user_id[train_rate == 2]]], 0)
            buy2_0 = torch.cat([dst_feature[train_item_id[train_rate == 1]], dst_feature[train_item_id[train_rate == 2]]], 0)
            buy_y_0 = torch.einsum('ij,kj->i', buy1_0 * buy2_0, eee)
            age_0 = torch.mean(buy_y_0)
            min_0 = torch.min(buy_y_0)
            max_0 = torch.max(buy_y_0)

            age_1 = torch.mean(rate_y)
            min_1 = torch.min(rate_y)
            max_1 = torch.max(rate_y)

            print()
            print('epoch:{:.4f}   loss:{:.4f}  n3:{:.4f}    n5:{:.4f}  '.format(epoch + 1, loss, mse, mae))
            print('负样本平均:{:.4f}   负样本min:{:.4f}   负样本max:{:.4f}   '.format(age_0, min_0, max_0))
            print('正样本平均:{:.4f}   正样本min:{:.4f}   正样本max:{:.4f}   '.format(age_1, min_1, max_1))
            print()

    # torch.save(model.state_dict(), 'model_dict.pth')
    # torch.save(src_feature, 'src_feature.pth')
    # torch.save(dst_feature, 'dst_feature.pth')