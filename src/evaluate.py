import numpy as np

from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR

from torch.utils.data import DataLoader
from torchvision.transforms import Resize

from data.dataset import Dataset
from models.icscn import ICSCN
from utils.options import parse_test_options

if __name__ == '__main__':
    opt = parse_test_options() 

    dataset = Dataset(opt.data_dir, opt.g_truth_dir, [Resize((256, 256))])
    dataloader = DataLoader(dataset,
                            shuffle=False,
                            num_workers=0,
                            batch_size=1)

    model = ICSCN()

    # Load the latest model saved
    model.load(0, opt.checkpoint_dir) 
    model.model.eval()

    ssim_metric = SSIM(data_range=1.0, reduction=None)
    psnr_metric = PSNR(data_range=1.0)

    ssim = []
    psnr = []
    for i, (rainy_image, clear_image) in enumerate(dataloader):
        derained_image = model(rainy_image).detach()
        
        ssim += [ssim_metric(rainy_image, clear_image)]
        psnr += [psnr_metric(rainy_image, clear_image)]
        
    ssim_std = np.std(ssim)
    psnr_std = np.std(psnr)

    ssim = np.mean(ssim)
    psnr = np.mean(psnr)
 
    print(ssim)
    print(ssim_std)

    print(psnr)
    print(psnr_std)
