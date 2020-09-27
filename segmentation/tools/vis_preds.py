import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
from PIL import Image

main_dir = sys.argv[1]
plt.figure(1); plt.figure(2); # Img overlay Gt, Pred
plt.figure(3);plt.figure(4); # Heatmaps
plt.figure(5); # Sprt Img overlay GT
plt.ion()
plt.show()

def PIL2array(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 4)

def create_overlay(img, mask, colors):
    im= Image.fromarray(np.uint8(img))
    im= im.convert('RGBA')

    mask_color= np.zeros((mask.shape[0], mask.shape[1],3))
    if len(colors)==3:
        mask_color[mask==colors[1],0]=255
        mask_color[mask==colors[1],1]=255
        mask_color[mask==colors[2],0]=255
    else:
        mask_color[mask==colors[1],2]=255

    overlay= Image.fromarray(np.uint8(mask_color))
    overlay= overlay.convert('RGBA')

    im= Image.blend(im, overlay, 0.7)
    blended_arr= PIL2array(im)[:,:,:3]
    img2= img.copy()
    img2[mask==colors[1],:]= blended_arr[mask==colors[1],:]
    return img2

for f in sorted(os.listdir(main_dir+'gt/')):
    print(f)
    gt = cv2.imread(main_dir+'gt/'+f, 0)
    pred = cv2.imread(main_dir+'pred/'+f, 0)
    img = cv2.imread(main_dir+'qry_images/'+f)[:, :, ::-1]
    hmaps_bg = cv2.imread(main_dir+'hmaps_bg/'+f, 0)
    hmaps_fg = cv2.imread(main_dir+'hmaps_fg/'+f, 0)
    sprt_img = cv2.imread(main_dir+'sprt_images/'+f.split('.')[0]+'_shot0.png')[:, :, ::-1]
    sprt_gt = cv2.imread(main_dir+'sprt_gt/'+f.split('.')[0]+'_shot0.png', 0)

    gt[gt!=1]=0
    gt[gt==1]=255
    overlay_qry_gt= create_overlay(img, gt, [0,255])

    pred[pred!=1]=0
    pred[pred==1]=255
    overlay_qry_pred= create_overlay(img, pred, [0,255])

    sprt_gt[sprt_gt!=16]=0
    sprt_gt[sprt_gt==16]=255
    overlay_sprt= create_overlay(sprt_img, sprt_gt, [0,255])

    plt.figure(1); plt.imshow(overlay_qry_gt); plt.figure(2); plt.imshow(overlay_qry_pred);
    plt.figure(3); plt.imshow(hmaps_bg); plt.figure(4); plt.imshow(hmaps_fg)
    plt.figure(5); plt.imshow(overlay_sprt);

    plt.draw()
    plt.waitforbuttonpress(0)

