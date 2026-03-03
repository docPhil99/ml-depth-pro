from PIL import Image
import depth_pro
import numpy as np
import matplotlib.pyplot as plt
image_path = "/home/phil/datasets/MOT/MOT17/train/MOT17-02-DPM/img1/000001.jpg"
# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Load and preprocess an image.
image, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image)

# Run inference.
prediction = model.infer(image, f_px=f_px)
depth_t = prediction["depth"]  # Depth in [m].
focallength_px = prediction["focallength_px"]  # Focal length in pixels.
print(f'Predicted focal length: {focallength_px}')
plt.figure()
depth = depth_t.cpu().numpy()
np.clip(depth, 0, 10, out=depth)
#depth.Tensor.ndim = property(lambda self: len(self.shape))
plt.imshow(depth)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

(x, y) = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
ax.plot_surface(x,y,depth)
plt.show()