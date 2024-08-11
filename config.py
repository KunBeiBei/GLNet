import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', nargs='?', default='Beibei', help='Choose a dataset from {Beibei,Taobao}')

    parser.add_argument('--epoch', type=int, default=4200, help='Number of epoch.')  # 4129
    parser.add_argument('--embed_size', type=int, default=64, help='Embedding size.')
    parser.add_argument('--layer_size', type=int, default=4, help='Output sizes of every layer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--Ks', nargs='?', default='[10, 50, 100]', help='K for Top-K list')

    # BeiBei parameters
    parser.add_argument('--MTL_BeiBei_w', nargs='?', default="{'pv':0.0/6, 'cart':5.0/6, 'buy':1.0/6}",
                        help='Regularization, [0.0/6, 5.0/6, 1.0/6] for beibei, [1.0/6, 4.0/6, 1.0/6] for taobao')
    parser.add_argument('--BeiBei_decay', type=float, default=1e-2,
                        help='Regularization, 1e-2 for beibei, x for taobao')

    parser.add_argument('--BeiBei_c0', type=float, default=600.0, help='')
    parser.add_argument('--BeiBei_i', type=float, default=0.5, help='')

    # Taobao parameters
    parser.add_argument('--MTL_TaoBao_w', nargs='?', default="{'pv':1.0/6, 'cart':4.0/6, 'buy':1.0/6}",
                        help='Regularization, [1.0/6, 4.0/6, 1.0/6] for beibei, [1.0/6, 4.0/6, 1.0/6] for taobao')
    parser.add_argument('--TaoBao_decay', type=float, default=1e-2,
                        help='Regularization, 1e-2 for beibei, x for taobao')

    parser.add_argument('--TaoBao_c0', type=float, default=300.0, help='')
    parser.add_argument('--TaoBao_i', type=float, default=0.5, help='')

    return parser.parse_args(args=[])
