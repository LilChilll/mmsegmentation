import torch
import torch.nn as nn
import torch.nn.functional as F

def reconstruct_visual_features(feature_map, mask_pred):
    """
    重构类别级别视觉特征
    :param feature_map: (B, D, H, W) 分割网络输出的特征图
    :param mask_pred: (B, C, H, W) 掩码预测，表示每个类别的概率分布
    :return: (B, C, D) 类别级别视觉表征
    """
    B, D, H, W = feature_map.shape
    _, C, _, _ = mask_pred.shape

    # 对 mask_pred 进行 softmax 归一化 (可选，确保 mask_pred 代表权重)
    mask_pred = F.softmax(mask_pred, dim=1)  # 按类别维度进行 softmax

    # 调整 feature_map 形状以便矩阵运算
    feature_map = feature_map.view(B, D, H * W)  # (B, D, H * W)
    mask_pred = mask_pred.view(B, C, H * W)     # (B, C, H * W)

    # 计算类别级别视觉表征
    # weighted_features[b, c, :] = sum(mask_pred[b, c, h, w] * feature_map[b, :, h, w])
    visual_features = torch.bmm(mask_pred, feature_map.permute(0, 2, 1))  # (B, C, D)

    return visual_features

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, visual_features, text_embeddings):
        """
        计算对比损失
        :param visual_features: (B, C, D) 类别级别视觉特征
        :param text_embeddings: (B, C, D) 类别级别文本嵌入
        :return: 对比损失
        """
        B, C, D = visual_features.shape

        # L2 归一化
        visual_features = F.normalize(visual_features, dim=-1)  # (B, C, D)
        text_embeddings = F.normalize(text_embeddings, dim=-1)  # (B, C, D)

        # 计算相似度矩阵
        similarity = torch.einsum('bcd,bcd->bc', visual_features, text_embeddings)  # (B, C)

        # 正样本对的相似度
        positive_similarity = torch.exp(similarity / self.temperature)  # (B, C)

        # 所有负样本对的相似度
        all_similarity = torch.exp(torch.einsum('bcd,bkd->bck', visual_features, text_embeddings) / self.temperature)  # (B, C, C)
        negative_similarity = all_similarity.sum(dim=-1) - positive_similarity  # (B, C)

        # 对比损失
        loss = -torch.log(positive_similarity / (positive_similarity + negative_similarity))  # (B, C)
        return loss.mean()

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha  # 用于掩码分割损失的权重
        self.beta = beta    # 用于对比损失的权重
        self.focal_loss = nn.CrossEntropyLoss()  # 使用交叉熵作为基础分割损失
        self.contrastive_loss = ContrastiveLoss()

    def forward(self, mask_pred, mask_gt, visual_features, text_embeddings):
        """
        计算多任务损失
        :param mask_pred: (B, C, H, W) 掩码预测
        :param mask_gt: (B, H, W) 掩码标签
        :param visual_features: (B, C, D) 类别级别视觉特征
        :param text_embeddings: (B, C, D) 类别级别文本嵌入
        :return: 总损失
        """
        # 计算分割任务的 Focal Loss
        seg_loss = self.focal_loss(mask_pred, mask_gt)

        # 计算对比学习损失
        contrast_loss = self.contrastive_loss(visual_features, text_embeddings)

        # 总损失
        total_loss = self.alpha * seg_loss + self.beta * contrast_loss
        return total_loss

# 示例用法
def example_usage():
    # 假设输入数据
    B, C, D, H, W = 2, 5, 512, 32, 32
    feature_map = torch.randn(B, D, H, W)  # (B, D, H, W)
    mask_pred = torch.randn(B, C, H, W)   # (B, C, H, W)
    mask_gt = torch.randint(0, C, (B, H, W))  # (B, H, W)
    text_embeddings = torch.randn(B, C, D)  # (B, C, D)

    # 重构视觉特征
    visual_features = reconstruct_visual_features(feature_map, mask_pred)

    # 初始化多任务损失函数
    multitask_loss_fn = MultiTaskLoss(alpha=1.0, beta=1.0)

    # 计算总损失
    total_loss = multitask_loss_fn(mask_pred, mask_gt, visual_features, text_embeddings)
    print("Total Loss:", total_loss.item())

if __name__ == "__main__":
    example_usage()
