import os
from mmseg.models.decode_heads.utils import mask_to_onehot, onehot_to_binary_edges
import mmcv
import numpy as np
from mmcv.utils import Config, DictAction, get_git_hash
from mmseg.datasets import build_dataset,build_dataloader
import torch
import matplotlib.pyplot as plt
from PIL import Image
# test_mask_path = os.path.join("/mnt/ssd/home/jcheng/EfficientNet/datasets/Pascal_VOC/VOCdevkit/VOC2012","SegmentationClass" ,'2007_000032.png')
image_root = "/mnt/ssd/home/jcheng/EfficientNet/datasets/Pascal_VOC/VOCdevkit/VOC2012/JPEGImages"
# mask = mmcv.imread(test_mask_path, flag='unchanged')
# gt_cls =np.unique(mask)
# print(gt_cls)
cfg_file = "configs/zegclip/sszegclip-20k_voc-512x512.py"

cfg = Config.fromfile(cfg_file)
cfg.data.samples_per_gpu = 1
cfg.data.workers_per_gpu = 1 
dataset = build_dataset(cfg.data.train)
palette = dataset.PALETTE
loader_cfg = dict(
   # cfg.gpus will be ignored if distributed
   num_gpus=1,
   dist=False,
   drop_last=True,
   shuffle=False)
# The overall dataloader settings
loader_cfg.update({
   k: v
   for k, v in cfg.data.items() if k not in [
      'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
      'test_dataloader'
   ]
})
val_loader_cfg = {**loader_cfg,**cfg.data.get('val_dataloader', {})}
data_loader = build_dataloader(dataset, **val_loader_cfg)
loader_indices = data_loader.batch_sampler
i = 0
target_gt_masks = []
for batch_indices, data in zip(loader_indices, data_loader):
   gt_semantic_seg = data['gt_semantic_seg'].data[0]
   target_gt_masks.append({
      'image_name':data['img_metas'].data[0][0]['ori_filename'],
      'img':data['img'].data[0],
      'mask':gt_semantic_seg[0]}) 
   if len(target_gt_masks)>5:
       break

# print(target_gt_mask.shape)
out_dir = 'edge_output'
if not os.path.exists(out_dir):
   os.makedirs(out_dir)

for target_gt_mask in target_gt_masks:
   f, axarr = plt.subplots(1,3,figsize=(24,6))
   for ax in axarr:
      ax.set_xticks([])
      ax.set_yticks([])
   mask = target_gt_mask['mask'] # 1,512,512
   mask = mask.squeeze(0)
   raw_image = target_gt_mask['img']
   raw_image = raw_image.squeeze(0).cpu().numpy()
   raw_image = raw_image.transpose(1,2,0)
   # reverse normalization
   mean = [123.675, 116.28, 103.53]
   std = [58.395, 57.12, 57.375]
   raw_image = raw_image*std+mean
   raw_image = raw_image.astype(np.uint8)
   image_name = target_gt_mask['image_name']
   out_path = os.path.join(out_dir,image_name)
   # out_raw_path = os.path.join(out_dir,image_name[:-4]+'_raw.png')
   # out_mask_path = os.path.join(out_dir,image_name[:-4]+'_mask.png')
   gt_cls = mask.unique()
   gt_cls = gt_cls[gt_cls !=255]
   print(gt_cls)
   _edgemap = mask.clone()
   _edgemap = _edgemap.cpu().numpy() # 512,512
   _edgemap = mask_to_onehot(_edgemap, gt_cls.clone().cpu().numpy())
   if len(_edgemap)>0:
      _edgemap = onehot_to_binary_edges(_edgemap, 2, gt_cls) # 1,512,512
      _edgemap = _edgemap*255
      # save the edge map
      # mmcv.imwrite(_edgemap.squeeze(0), out_edge_path)
      _edgemap = Image.fromarray(_edgemap.squeeze(0))
      axarr[0].imshow(_edgemap)
      # save the mask using palette
      color_seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
      for label, color in enumerate(palette):
         color_seg[mask == label, :] = color
      color_seg = color_seg[..., ::-1]
      # mmcv.imwrite(color_seg, out_mask_path)
      color_seg = Image.fromarray(color_seg)
      axarr[1].imshow(color_seg)
      # save the raw image
      # mmcv.imwrite(raw_image, out_raw_path)
      axarr[2].imshow(raw_image)

   plt.tight_layout()
   plt.savefig(out_path)
   plt.close()



      # edge_map = torch.from_numpy(_edgemap).float().squeeze(0)  # 512,512
      # edge_gts = edge_map.cuda() # (32,32)
