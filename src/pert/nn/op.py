import numpy as np
import torch

from torch.autograd import Function
from ...model.op import look_up_category_embedding


@torch.inference_mode()
def compute_pert_transition(x, pert_ind, pert_vec, pert_val, grad_output):
    """Compute transition matrix for perturbate vector

    :param x: input feature
    :param pert_ind: perturbate vector [pert_num,]
    :param pert_vec: perturbate vector [1, gene_num]
    :param pert_val: perturbate value [1, gene_num]
    :param grad_output: gradient w.r.t pert_x [batch, gene_num]
    :return: computed transition matrix [pert_num, gene_num]
    """
    diff_pert = pert_val - x
    diff_pert_times_grad = diff_pert * grad_output

    # we replace perturbation gene so the original perturbation effect is removed
    sub_pert = torch.index_select(
        diff_pert_times_grad, dim=1, index=pert_ind
    ).transpose_(
        0, 1
    )  # [num_pert, batch]

    # only those genes don't be perturbated previously can have effects on adding
    # perturbations
    add_pert = diff_pert_times_grad * (1 - pert_vec)  # [batch, gene_num]
    appr_pert_trans = torch.mean(
        add_pert.unsqueeze(0) - sub_pert.unsqueeze(-1), dim=1
    )  # [num_pert, batch, gene_num] => [num_pert, gene_num]

    # since we subtract extra `sub_pert` on unchanged genes,
    # we need to correct them to zeros
    rows = torch.arange(0, appr_pert_trans.size(0), device=appr_pert_trans.device)
    cols = pert_ind
    appr_pert_trans[rows, cols] = 0.0

    return appr_pert_trans


@torch.inference_mode()
def compute_embedding_pert_transition(
    x, pert_ind, pert_vec, pert_val, grad_output, embed_weight
):
    """Compute transition matrix for perturbate vector

    :param x: input feature [batch, gene_num]
    :param pert_ind: perturbate vector [pert_num,]
    :param pert_vec: perturbate vector [1, gene_num]
    :param pert_val: perturbate value [1, gene_num]
    :param grad_output: gradient w.r.t the embedding of pert_x [batch, gene_num,
                        embd_dim]
    :param embed_weight: category embedding weight [gene_num, category_num, embed_dim]
    :return: computed transition matrix [pert_num, gene_num]
    """
    diff_pert = look_up_category_embedding(
        pert_val, embed_weight
    ) - look_up_category_embedding(x, embed_weight)
    diff_pert_times_grad = torch.sum(diff_pert * grad_output, dim=-1)

    # we replace perturbation gene so the original perturbation effect is removed
    sub_pert = torch.index_select(
        diff_pert_times_grad, dim=1, index=pert_ind
    ).transpose_(
        0, 1
    )  # [num_pert, batch]

    # only those genes don't be perturbated previously can have effects on adding
    # perturbations
    add_pert = diff_pert_times_grad * (1 - pert_vec)  # [batch, gene_num]
    appr_pert_trans = torch.mean(
        add_pert.unsqueeze(0) - sub_pert.unsqueeze(-1), dim=1
    )  # [num_pert, batch, gene_num] => [num_pert, gene_num]

    # since we subtract extra `sub_pert` on unchanged genes,
    # we need to correct them to zeros
    rows = torch.arange(0, appr_pert_trans.size(0), device=appr_pert_trans.device)
    cols = pert_ind
    appr_pert_trans[rows, cols] = 0.0

    return appr_pert_trans


def create_pert_vec(pert_ind: torch.LongTensor, length: int):
    """Create perturbation vector from index

    :param pert_ind: one dimensional index tensor
    :param length: the total length of perturbation vector
    :return: created perturbation vector
    """
    if pert_ind.dim() != 1:
        raise Exception("Perturbation index should be only one dimension.")
    pert_vec = pert_ind.new_zeros((length,)).float()
    pert_vec.index_fill_(dim=0, index=pert_ind, value=1.0)
    return pert_vec


def create_trans_pert_vecs(pert_vec: torch.FloatTensor, trans_inds: torch.LongTensor):
    """Create a batch of transisted perturbation vectors beased on perturbation index

    :param pert_vec: origin perturbation vector [pert_size,]
    :param trans_inds: updated perturbation index [pert_num, topk]
    :return: a batch of transisted perturbation vectors [pert_num * topk, pert_size]
    """
    top_k = trans_inds.size(1)
    trans_nums = trans_inds.numel()

    # get perturbated indexes and repeat them for top_k times
    pert_inds = torch.nonzero(pert_vec, as_tuple=True)[0]

    if pert_inds.size(0) != trans_inds.size(0):
        raise ValueError("Current perturbation number mismatch with the transition.")

    pert_inds = pert_inds.repeat_interleave(top_k)
    trans_pert_vecs = torch.tile(pert_vec, (trans_nums, 1))
    rows = torch.arange(0, trans_pert_vecs.size(0), device=trans_pert_vecs.device)
    # close all old perturbated positions
    trans_pert_vecs[rows, pert_inds] = 0
    # open all transisted postions
    trans_pert_vecs[rows, trans_inds.flatten()] = 1

    return trans_pert_vecs


def create_trans_pert_vecs_from_ind(
    pert_ind: torch.LongTensor, trans_inds: torch.LongTensor, length: int
):
    """Create a batch of transisted perturbation vectors beased on perturbation index

    :param pert_ind: origin perturbation index [pert_num,]
    :param trans_inds: updated perturbation index [pert_num, topk]
    :return: a batch of transisted perturbation vectors [pert_num * topk, pert_size]
    """
    top_k = trans_inds.size(1)
    trans_nums = trans_inds.numel()

    if pert_ind.size(0) != trans_inds.size(0):
        raise ValueError("Current perturbation number mismatch with the transition.")

    pert_vec = create_pert_vec(pert_ind, length).to(pert_ind.device)
    pert_ind = pert_ind.repeat_interleave(top_k)
    trans_pert_vecs = torch.tile(pert_vec, (trans_nums, 1))
    rows = torch.arange(0, trans_pert_vecs.size(0), device=pert_ind.device)
    # close all old perturbated positions
    trans_pert_vecs[rows, pert_ind] = 0
    # open all transisted postions
    trans_pert_vecs[rows, trans_inds.flatten()] = 1

    return trans_pert_vecs


def initial_random_ind(num, total_length):
    inds = list(range(total_length))
    choiced_inds: np.ndarray = np.random.choice(inds, size=num, replace=False)
    return choiced_inds.tolist()


class BatchTriggerPerturbationFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.FloatTensor,
        pert_vec: torch.FloatTensor,
        pert_val: torch.FloatTensor,
    ):
        """_summary_

        :param ctx: funtion context
        :param x: input sequence [batch, gene_num]
        :param pert_vec: binary float vectors indicating perturbating
               genes [trial_num, gene_num]
        :param pert_val: a float vector indicating perturbating gene expressions
        [gene_num]
        :return: perturbated input sequence [trial_num * batch, gene_num]
        """
        x = x.unsqueeze(0)
        pert_vec = pert_vec.unsqueeze(1)
        pert_val = pert_val[None, None, :]

        length = pert_val.shape[-1]
        pert_x = (
            x + (pert_val - x) * pert_vec
        )  # x * (1 - pert_vec) + pert_vec * pert_val
        flat_pert_x = pert_x.reshape((-1, length))
        return flat_pert_x

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor):
        raise Exception("Batch inference function cannot do backward.")


class SequenceTriggerPerturbationFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.FloatTensor,
        pert_ind: torch.LongTensor,
        pert_val: torch.FloatTensor,
        trans_mx: torch.FloatTensor,
    ):
        """Perturbate the input sequence with perturbate mask and values

        :param ctx: funtion context
        :param x: input sequence [batch, gene_num]
        :param pert_vec: a binary float vector indicating perturbating
               genes [gene_num, ]
        :param pert_val: a float vector indicating perturbating gene expressions
        :param trans_mx: a transition matrix placeholder for pert_vec
        :return: perturbated input sequence [batch, gene_num]
        """
        length = pert_val.shape[-1]
        pert_vec = create_pert_vec(pert_ind, length=length).to(x.device)
        ctx.save_for_backward(x, pert_ind, pert_vec, pert_val)

        pert_vec = pert_vec.unsqueeze(0)
        pert_val = pert_val.unsqueeze(0)
        pert_x = (
            x + (pert_val - x) * pert_vec
        )  # x * (1 - pert_vec) + pert_vec * pert_val
        return pert_x

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor):
        """Backward to get update information
        !!! This is not a traditional backward compatible funciton

        Pls note that, for pertubate vector `pert_vec`, we compute the transition
        matrix instead, which indicates the approximated target value changes caused by
        replaceing one perturbated gene with other genes

        :param ctx: function context
        :param grad_output: the gradient w.r.t pert_x
        :return: computed gradient or transition matrix for each term
        """
        x, pert_ind, pert_vec, pert_val = ctx.saved_tensors
        pert_vec = pert_vec.unsqueeze(0)
        pert_val = pert_val.unsqueeze(0)

        grad_x: torch.Tensor = grad_output * (1 - pert_vec)
        grad_pert_val = torch.sum(grad_output * pert_vec, dim=0)
        tras_pert_vec = compute_pert_transition(
            x, pert_ind, pert_vec, pert_val, grad_output
        )
        return grad_x, None, grad_pert_val, tras_pert_vec


class SequenceEmbeddingTriggerPerturbationFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.FloatTensor,
        pert_ind: torch.LongTensor,
        pert_val: torch.FloatTensor,
        embed_weight: torch.FloatTensor,
        trans_mx: torch.FloatTensor,
    ):
        """Perturbate the input sequence with perturbate mask and values

        :param ctx: funtion context
        :param x: input sequence [batch, gene_num]
        :param pert_vec: a binary float vector indicating perturbating
               genes [gene_num, ]
        :param pert_val: a float vector indicating perturbating gene expressions
        :param embedding_weight: a float vector indicating embeddings for each category
                                 [gene_num, category_num, embd_dim]
        :param trans_mx: a transition matrix placeholder for pert_vec
        :return: perturbated input sequence [batch, gene_num]
        """
        length = pert_val.shape[-1]
        pert_vec = create_pert_vec(pert_ind, length=length).to(x.device)
        ctx.save_for_backward(x, pert_ind, pert_vec, pert_val, embed_weight)

        pert_vec = pert_vec.unsqueeze(0)
        pert_val = pert_val.unsqueeze(0)
        pert_x = (
            x + (pert_val - x) * pert_vec
        )  # x * (1 - pert_vec) + pert_vec * pert_val

        # get embedding of pert_x
        with torch.no_grad():
            pert_x_embed = look_up_category_embedding(pert_x.long(), embed_weight)
        pert_x_embed.requires_grad_()
        return pert_x_embed

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor):
        """Backward to get update information
        !!! This is not a traditional backward compatible funciton

        Pls note that, for pertubate vector `pert_vec`, we compute the transition
        matrix instead, which indicates the approximated target value changes caused by
        replaceing one perturbated gene with other genes

        :param ctx: function context
        :param grad_output: the gradient w.r.t pert_x
        :return: computed gradient or transition matrix for each term
        """
        x, pert_ind, pert_vec, pert_val, embed_weight = ctx.saved_tensors
        pert_vec = pert_vec.unsqueeze(0)
        pert_val = pert_val.unsqueeze(0)

        tras_pert_vec = compute_embedding_pert_transition(
            x, pert_ind, pert_vec, pert_val, grad_output, embed_weight
        )
        return None, None, None, None, tras_pert_vec
