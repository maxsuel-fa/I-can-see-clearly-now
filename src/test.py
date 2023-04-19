from data.dataset import Dataset
from models.icscn import ICSCN
from utils.options import parse_train_options

from PIL import Image
from torchvision.transforms import ToPILImage

if __name__ == '__main__':
    opt = parse_test_options() 

    dataset = Dataset(opt.data_dir, opt.g_truth_dir)
    dataloader = DataLoader(dataset,
                            shuffle=False,
                            num_workers=0)

    model = ICSCN()

    # Load the latest model saved
    model.load(0, opt.checkpoint_dir) 
    
    # to convert tensor to PIL image
    to_pil = ToPILImage(mode='RGB')

    for i, (rainy_image, clear_image) in enumerate(dataloader):
        derained_image = model(rainy_image)
        derained_image = to_pil(derained_image)

        path = os.path.join(save_dir, 'derained_image' + str(i))
        derained_image.save(path)
