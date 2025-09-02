import cv2
from depth import disp
import numpy as np
from PIL import Image

max_disp = 192

model = disp.my_load_model(r"G:\mobelnetV11v2\250403HCDatasets\best_3px.pth")
pred_disp = disp.get_disp(model,r"G:\mobelnetV11v2\alg\20250421_162253/20250421_162253_cam1_rect.png", r"G:\mobelnetV11v2\alg\20250421_162253/20250421_162253_cam2_rect.png",1280,736)
depth = (2873.41333122683 * 18.8689953791335 / 2) / pred_disp
# DEPTH= cv2.resize(depth,(3072,4096))
depth_img = Image.fromarray(depth.astype(np.uint16))
depth_img.save('1depth_img.png')

img = Image.fromarray((pred_disp * 256).astype(np.uint16))
img.save('1Test_disparity.png')

disparity_color = cv2.applyColorMap(cv2.convertScaleAbs(pred_disp, alpha=256 / max_disp), cv2.COLORMAP_JET)
cv2.imwrite('1disparity_color.png', disparity_color)