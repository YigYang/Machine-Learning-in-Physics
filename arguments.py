import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file_name', type=str, default="train_features.npy", help='train file name')
    parser.add_argument('--valid_file_name', type=str, default="valid_features.npy", help='valid file name')
    parser.add_argument('--test_file_name', type=str, default="test_features.npy", help='test file name')
    parser.add_argument('--label_file_name', type=str, default="labels.fits", help='label file name')
    # Linear regression Todo: Select best weight decay and run it on test
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--open_ridge', type=bool, default=True, help='weight decay open or not')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--weight_decay', type=float, default=0.0, help="0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2")
    # KNN (Warning: use batch to avoid crashing memory)
    parser.add_argument('--k_value', type=int, default=33, help='value around sqrt(1024)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    # MLP
    parser.add_argument('--h1', type=int, default=128, help='h1 value')
    parser.add_argument('--h2', type=int, default=64, help='h2 value')
    parser.add_argument('--lr_mlp', type=float, default=1e-3, help='learning rate decay')
    parser.add_argument('--epochs_mlp', type=int, default=200, help='mlp epochs')


    return parser.parse_args()