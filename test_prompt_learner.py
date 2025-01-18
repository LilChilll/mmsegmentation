from mmseg.models.backbones import DynamicPromptCLIPTextEncoder,MultiGranularityPromptLearner
import torch
CLASS_NAMES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor')
base_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
novel_class = [15, 16, 17, 18, 19]
both_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_encoder = DynamicPromptCLIPTextEncoder(tpt_mode=True)
    pretrained = r"/mnt/ssd/home/jcheng/EfficientNet/YOLOXX/checkpoints/ViT-B-16.pt"
    text_encoder.init_weights(pretrained)
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    # print(next(text_encoder.parameters()).device)
    # prompt_learner = CoOpPromptLearner(CLASS_NAMES, text_encoder,CSC=False)
    n_layers = 8
    # print(len(CLASS_NAMES))
    prompt_learner = MultiGranularityPromptLearner(CLASS_NAMES,
                                                   seen_idx=base_class,
                                                   all_idx=both_class,
                                                   N_CTX=8,
                                                   n_layers = n_layers)
    prompt_learner.init_context(text_encoder)
    prompt_learner = prompt_learner.to(device)
    image_features = n_layers*[torch.randn(4,1024,512).to(device)]
    text_embedding = prompt_learner(image_features,text_encoder)
    print(text_embedding.shape)
    