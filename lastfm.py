# 单行为+边
import tqdm
import torch
import numpy as np
from model import GLNID as GLN
from evaluate import get_ndcg


def test(test_model, test_user_number, test_item_number, test_edge_number, test_device):

    with (torch.no_grad()):

        test_model.eval()
        test_model = test_model.to(test_device)

        test_user = torch.load('data/' + dataset + '/test_user.pth')
        test_item = torch.load('data/' + dataset + '/test_item.pth')
        test_user = test_user.to(test_device)
        test_item = test_item.to(test_device)
        test_user_0 = torch.load('data/' + dataset + '/test_user_0.pth')
        test_item_0 = torch.load('data/' + dataset + '/test_item_0.pth')
        test_user_0 = test_user_0.to(test_device)
        test_item_0 = test_item_0.to(test_device)
        testset_user = torch.load('data/' + dataset + '/testset_user.pth')

        src_test, dst_test, edge_buy = test_model(test_user_number, test_item_number, test_edge_number)

        ndcg3 = []
        ndcg5 = []
        for u_id in testset_user:
            # item正负样本下标
            i_id1 = test_item[test_user == u_id]
            i_id0 = test_item_0[test_user_0 == u_id]
            i_id = torch.cat([i_id1, i_id0])
            dst = dst_test[i_id]
            # user特征
            src = src_test[u_id]
            src = src.repeat(i_id.shape[0], 1)
            # 真实值
            zy_1 = torch.tensor([1 for _ in range(i_id1.shape[0])]).to(test_device)
            zy_0 = torch.tensor([0 for _ in range(i_id0.shape[0])]).to(test_device)
            zy = torch.cat([zy_1, zy_0])
            # pre
            y = torch.einsum('ij,kj->i', src * dst, edge_buy)
            _, recommendation_list = y.view(-1).sort(descending=True)
            recommendation_list = recommendation_list.tolist()
            n3 = get_ndcg(recommendation_list, zy.tolist(), 3)
            n5 = get_ndcg(recommendation_list, zy.tolist(), 5)
            ndcg3.append(n3)
            ndcg5.append(n5)

        n3 = np.mean(ndcg3)
        n5 = np.mean(ndcg5)

        return n3, n5


if __name__ == '__main__':
    SEED = 2020
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Data loading...")

    # dataset = args.dataset
    dataset = 'lastfm_20'

    # embed size
    d = 64

    traNum1 = 0
    traNum2 = 25000
    lr = 1e-4
    decay = 1e-2

    train_user = torch.load('data/' + dataset + '/train_user.pth')
    train_item = torch.load('data/' + dataset + '/train_item.pth')
    train_user_0 = torch.load('data/' + dataset + '/train_user_0.pth')
    train_item_0 = torch.load('data/' + dataset + '/train_item_0.pth')

    Cv_f_buy = 0.001

    print("data Loading completed")

    num_user = 1872
    num_item = 3846

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model = GLN(num_user, num_item, d, device)

    # Total_params = 0.386752M
    # Trainable_params = 0
    # NonTrainable_params = 0
    # for param in model.parameters():
    #     mulValue = np.prod(param.size())
    #     Total_params += mulValue  # 总参数量
    #     if param.requires_grad:
    #         Trainable_params += mulValue  # 可训练参数量
    #     else:
    #         NonTrainable_params += mulValue  # 非可训练参数量
    #
    # print(f'Total params: {Total_params / 1e6}M')
    # print(f'Trainable params: {Trainable_params / 1e6}M')
    # print(f'Non-trainable params: {NonTrainable_params / 1e6}M')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mses = []
    maes = []

    # train & test
    for epoch in tqdm.tqdm(range(traNum1, 50000)):
        model.train()

        # Cv_f_buy = Cv_f_buy.to(device)

        model = model.to(device)
        optimizer.zero_grad()

        # number_user_train = torch.cat([train_data_idx, val_data_idx]).to(device)  # 1497
        user_number = torch.arange(1872).to(device)
        item_number = torch.arange(3846).to(device)
        edge_number = torch.arange(1).to(device)
        src_feature, dst_feature, eee = model(user_number, item_number, edge_number)

        temp = torch.einsum('ab,ac->bc', dst_feature, dst_feature) * torch.einsum('ab,ac->bc', src_feature, src_feature)

        buy1 = src_feature[train_user]
        buy2 = dst_feature[train_item]

        buy_y = torch.einsum('ij,kj->i', buy1 * buy2, eee)

        buy_temp = temp * torch.einsum('ab,ac->bc', eee, eee)

        loss1 = Cv_f_buy * torch.sum(buy_temp) + torch.sum((1 - Cv_f_buy) * torch.pow(buy_y, 2) - 2 * buy_y)

        reg = torch.norm(src_feature) ** 2 / 2 + torch.norm(dst_feature) ** 2 / 2
        l2_loss = reg * decay

        loss = loss1 + l2_loss

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        if (epoch+1) % 100 == 0:
            mse, mae = test(model, user_number, item_number, edge_number, torch.device('cuda'))

            buy1_0 = src_feature[train_user_0]
            buy2_0 = dst_feature[train_item_0]
            buy_y_0 = torch.einsum('ij,kj->i', buy1_0 * buy2_0, eee)

            age_0 = torch.mean(buy_y_0)
            min_0 = torch.min(buy_y_0)
            max_0 = torch.max(buy_y_0)
            age_1 = torch.mean(buy_y)
            min_1 = torch.min(buy_y)
            max_1 = torch.max(buy_y)
        #     mses.append(mse)
        #     maes.append(mae)
        #     print(buy_temp)
        #     print(float(loss), float(reg))
        #     print(float(mse), float(mae))
            print()
            print('epoch:{:.4f}   loss:{:.4f}  n3:{:.4f}    n5:{:.4f}  '.format(epoch + 1, loss, mse, mae))
            print('负样本平均:{:.4f}   负样本min:{:.4f}   负样本max:{:.4f}   '.format(age_0, min_0, max_0))
            print('正样本平均:{:.4f}   正样本min:{:.4f}   正样本max:{:.4f}   '.format(age_1, min_1, max_1))
            print()

    # torch.save(model.state_dict(), 'model_dict.pth')
    # torch.save(src_feature, 'src_feature.pth')
    # torch.save(dst_feature, 'dst_feature.pth')
