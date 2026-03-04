from PIL import Image
import depth_pro
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from loguru import logger
import torch

class Depth:
    def __init__(self):
        # Load model and preprocessing transform
        self.model, self.transform = depth_pro.create_model_and_transforms(
            device=self._get_device(),precision=torch.half)
        self.model.eval()
        logger.info('Loaded model')

    def _get_device(self):
        """Get the Torch device."""
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        logger.info(f'Using device: {device}')
        return device

    def infer(self, image_path: Path):
        # Load and preprocess an image.
        image, _, f_px = depth_pro.load_rgb(image_path)
        image = self.transform(image)
        # Run inference.
        prediction = self.model.infer(image, f_px=f_px)
        #depth_t = prediction["depth"]  # Depth in [m].
        focallength_px = prediction["focallength_px"]  # Focal length in pixels.
        logger.debug(f'Predicted focal length: {focallength_px}')
        return prediction



parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-i", "--image", type=Path, help = "Path to the image")
group.add_argument("-d",'--directory', type=Path, help = "Path to the directory")
parser.add_argument('--output_dir', type=Path, default='./outputs')
parser.add_argument('--display', action='store_true')
args = parser.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=True)
if args.image:
    image_paths = [args.image]
else:
    image_paths  = sorted(Path(args.directory).glob('*.jpg'))
    logger.info(f'Found {len(image_paths)} images')
depth_model = Depth()
for image_path in image_paths:
    out_file = args.output_dir / Path(image_path.name).with_suffix('.npz')
    logger.debug(f'Output file: {out_file}')
    #image_path = "/home/phil/datasets/MOT/MOT17/train/MOT17-04-DPM/img1/000001.jpg"


    prediction = depth_model.infer(image_path)
    torch.save(prediction, out_file)
    depth_t, focal_length = prediction["depth"], prediction["focallength_px"]
    if args.display:
        plt.figure()
        depth = depth_t.cpu().numpy()
        #np.clip(depth, 0, 10, out=depth)
        #depth.Tensor.ndim = property(lambda self: len(self.shape))

        plt.imshow(depth)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        (x, y) = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        ax.plot_surface(x,y,depth)
        plt.show()