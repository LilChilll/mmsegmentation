# create venv
using python3.9 or python3.8
# install pytorch
```bash
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```
# install mmcv
```bash
    pip install -U openmim
    mim install mmengine==0.10.5
    mim install mmcv-full==1.4.4
    pip install mmdet==3.3.0 #(optional)
``` 
# install mmsegmentation
```bash
    pip install -v -e .
```
# other requirments
```bash
    pip install regex ftfy yapf==0.40.1 timm==0.3.2
    pip install numpy==1.24.4
```
# modify site-package
```python
    # path: venv/lib/python3.9/site-packages/timm/models/layers/
    # replace：
    from torch._six import container_abcs
    # with:
    import torch
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])
    if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
        from torch._six import container_abcs
    else:
        import collections.abc as container_abcs

```
# train model scripts
config file list:
    configs/zegclip/sszegclip-20k_voc-512x512_zero_dppt.py
    configs/zegclip/sszegclip-20k_voc-512x512_zero_dppt_shape.py
    configs/zegclip/sszegclip-20k_voc-512x512_zero_dppt_PL.py
    configs/zegclip/sszegclip-20k_voc-512x512_zero_dppt_PL_shape.py
    configs/zegclip/sszegclip-20k_voc-512x512_zero_dppt_plain.py
    configs/zegclip/sszegclip-20k_voc-512x512_zero_dppt_vpt.py
```bash
# use all gpus
bash tools/dist_train.sh configs/zegclip/sszegclip-20k_voc-512x512.py 2 --work-dir work_dirs/run1
# specify gpus
CUDA_VISIBLE_DEVICES="1,2" bash tools/dist_train.sh configs/zegclip/sszegclip-20k_voc-512x512.py 2 --work-dir work_dirs/run1_zegclip_voc_zero
```
# test model scripts
```bash
# use all gpus
python tools/test.py configs/zegclip/sszegclip-20k_voc-512x512.py work_dirs/run1_zegclip_voc_zero/iter_20000.pth --eval mIoU
```