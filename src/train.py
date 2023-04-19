from data.dataset import Dataset
from models.icscn import ICSCN
from utils.options import parse_train_options

if __name__ == '__main__':
    opt = parse_train_options()
    dataset = Dataset(opt.data_dir, opt.g_truth_dir)
    model = ICSCN(is_train=True)
    model.train(dataset, opt)
