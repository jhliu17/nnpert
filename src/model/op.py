import torch
import torch.nn.functional as F


def look_up_category_embedding(x: torch.LongTensor, embed_weight: torch.FloatTensor):
    """Look up embedding for each category

    :param x: input sequence [batch, gene_num]
    :param embed_weight: category embedding weight [gene_num, category_num, embd_dim]
    :return: embedding for input sequence [batch, gene_num, embd_dim]
    """
    batch_size = x.size(0)
    embed_dim = embed_weight.size(-1)

    weight = embed_weight.unsqueeze(0).expand(batch_size, -1, -1, -1)
    ind = x.unsqueeze(-1).expand(-1, -1, embed_dim).unsqueeze(-2).long()
    embedding = torch.gather(weight, dim=2, index=ind)
    return embedding.squeeze(2)


def balanced_sampling(x_true, threshold: float = 1e-4):
    pos_mask = x_true > threshold
    pos_nums = pos_mask.float().sum()

    # check whether the threshold setting is failed
    if pos_nums == x_true.numel():
        raise Exception(
            f"With current threshold ({threshold}), all samples are positive."
            f"The maximum value in the sample is {x_true.max()}."
        )

    # check whether the threshold setting contains imbalance problem
    pos_prob = pos_nums / x_true.numel()
    if pos_prob > 0.5:
        raise Warning(
            f"With current threshold ({threshold}), there are more than"
            f"half of samples are masked as positives."
        )

    neg_prob = pos_nums / (x_true.numel() - pos_nums)
    binomial = torch.distributions.Binomial(1, torch.ones_like(x_true) * neg_prob)
    neg_sample_mask = binomial.sample() == 1
    neg_mask = torch.logical_and(torch.logical_not(pos_mask), neg_sample_mask)
    return pos_mask, neg_mask


def masked_mse(x_pred, x_true, eps: float = 1e-4):
    """Masked MSE Loss
    we only compute the balance loss from the postive value and the negative value.
    """
    pos_mask, neg_mask = balanced_sampling(x_true, eps)

    mse_loss = F.mse_loss(x_pred, x_true, reduction="none")
    pos_loss = (mse_loss * pos_mask.float()).sum()
    neg_loss = (mse_loss * neg_mask.float()).sum()
    loss = (pos_loss + neg_loss) / (pos_mask.float().sum() + neg_mask.float().sum())
    return loss


def focal_bce_with_logits(
    x_pred: torch.Tensor, x_true: torch.Tensor, threshold: float, gamma: float = 0.0
):
    """compute bce loss with logits (ref: https://paperswithcode.com/method/focal-loss#:~:text=Focal%20loss%20applies%20a%20modulating,in%20the%20correct%20class%20increases.)

    :param x_pred: prediction logits
    :param x_true: binary ground truth
    :param threshold: threshold for determining postive class
    :param gamma: focal loss gamma, defaults to 0.
    :return: focal loss
    """
    x_pred_dist = torch.sigmoid(x_pred)
    pos_cls_mask = x_true > threshold
    neg_cls_mask = torch.logical_not(pos_cls_mask)

    bce_loss = F.binary_cross_entropy_with_logits(x_pred, x_true, reduction="none")

    bce_loss = (
        torch.pow(1 - x_pred_dist, gamma) * bce_loss * pos_cls_mask.float()
        + torch.pow(x_pred_dist, gamma) * bce_loss * neg_cls_mask.float()
    )
    return bce_loss


def masked_bce(x_pred, x_true, gamma: float = 1):
    """Masked BCE Loss
    we only compute the balance loss from the postive value and the negative value.
    """
    pos_mask, neg_mask = balanced_sampling(x_true, 0.5)

    bce_loss = focal_bce_with_logits(x_pred, x_true, threshold=0.5, gamma=gamma)
    pos_loss = (bce_loss * pos_mask.float()).sum()
    neg_loss = (bce_loss * neg_mask.float()).sum()

    """
    Note:

    The correct loss considering both the positive part and the negative part should be
        loss = (pos_loss + neg_loss) / (pos_mask.float().sum() + neg_mask.float().sum())
    where positive samples and negative samples are weighted the same for training.
    However, in practice, it doesn't work. The inituition reason is negative samples are
    much more easier to predict than positive samples. Therefore, we need to give
    different weights for positive samples and negative samples.

    We can compute the weighted loss as:
        loss = (pos_loss + neg_loss) / (pos_mask.float().sum() + neg_loss.float().sum())
    or
        use the focal bce loss?
    """
    loss = (pos_loss + neg_loss) / (pos_mask.float().sum() + neg_mask.float().sum())
    return loss
