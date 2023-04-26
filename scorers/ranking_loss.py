import torch as _torch
import torch
from torch import nn
import numpy as np
from functools import partial


def first_egg_loss(predicted_scores, true_scores, temperature=1.0):
    # logits = predicted_scores / temperature
    # y = nn.functional.gumbel_softmax(logits, tau=1.0, hard=True)
    pred_max_score, pred_max_indices = torch.max(predicted_scores, dim=1)
    pred_max_true_score = torch.gather(true_scores, 1, pred_max_indices.unsqueeze(1))
    true_better_scores = (true_scores > pred_max_true_score+0.1).int()
    true_worse_scores = (true_scores < pred_max_true_score-0.1).int()
    # pred_max_scores = (predicted_scores * true_scores).sum(dim=1)
    # print (pred_max_scores
    # true_better_scores = true_scores >= pred_max_scores
    gap = pred_max_score.unsqueeze(-1) - predicted_scores
    better_loss =  true_better_scores * (gap + 0.2)
    # worse_loss = -(true_worse_scores * gap)
    # loss = better_loss + worse_loss
    loss = better_loss
    # Better loss, we encourage the score of better samples to increase
    loss = torch.mean(loss)
    # true_max_scores, _ = torch.max(true_scores, dim=1)
    # loss = torch.mean(torch.square(true_max_scores - pred_max_scores))
    return loss
    

def pairwise_hinge_loss(scores, relevance, margin=1.0):
    """Computes the Pairwise Hinge Loss.
    The Pairwise Hinge Loss measures the degree of difference between the
    relevance scores of a pair of documents. The loss is zero if the difference
    between the relevance scores is greater than or equal to a margin.
    Args:
        scores: A tensor of shape (batch_size, list_size) containing the
            predicted scores for each document.
        relevance: A tensor of shape (batch_size, list_size) containing the
            relevance labels for each document.
        margin: The margin value.
    Returns:
        A tensor of shape (batch_size,) containing the loss for each sample.
    """
    score_pairs = batch_pairs(scores)
    rel_pairs = batch_pairs(relevance)
    score_pair_diffs = score_pairs[:, :, :, 0] - score_pairs[:, :, :, 1]
    rel_pair_diffs = rel_pairs[:, :, :, 0] - rel_pairs[:, :, :, 1]
    loss = margin - score_pair_diffs
    loss = _torch.clamp(loss, min=0.0)
    loss[rel_pair_diffs <= 0.1] = 0.0
    return loss.mean()

def batch_pairs(x: _torch.Tensor) -> _torch.Tensor:
    """Returns a pair matrix
    This matrix contains all pairs (i, j) as follows:
        p[_, i, j, 0] = x[_, i]
        p[_, i, j, 1] = x[_, j]
    Args:
        x: The input batch of dimension (batch_size, list_size) or
            (batch_size, list_size, 1).
    Returns:
        Two tensors of size (batch_size, list_size ^ 2, 2) containing
        all pairs.
    """

    if x.dim() == 2:
        x = x.reshape((x.shape[0], x.shape[1], 1))

    # Construct broadcasted x_{:,i,0...list_size}
    x_ij = _torch.repeat_interleave(x, x.shape[1], dim=2)

    # Construct broadcasted x_{:,0...list_size,i}
    x_ji = _torch.repeat_interleave(x.permute(0, 2, 1), x.shape[1], dim=1)

    return _torch.stack([x_ij, x_ji], dim=3)

def kl_divergence_loss(pred_scores, true_scores, temperature=1.0, eps=1e-6):
    log_pred_scores = _torch.log_softmax(pred_scores / temperature, dim=1)
    log_pred_scores = log_pred_scores - log_pred_scores.logsumexp(dim=-1, keepdim=True)
    # true_scores = _torch.nn.functional.softmax(true_scores, dim=1)
    true_scores_sum = true_scores.sum(dim=1, keepdim=True)
    # true_scores_sum = torch.clamp(true_scores_sum, min=eps)
    true_scores = true_scores / true_scores_sum
    true_scores = _torch.clamp(true_scores, min=eps, max=1.0 - eps)
    loss = _torch.nn.KLDivLoss(reduction='batchmean')(log_pred_scores, true_scores)
    return loss


def tau_sigmoid(tensor, tau, target=None, general=None):
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it
    through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / tau
    # clamp the input tensor for stability
    exponent = 1. + exponent.clamp(-50, 50).exp()
    return 1.0 / exponent



class SmoothRankAP(nn.Module):
    def __init__(
        self,
        rank_approximation,
        return_type='1-mAP',
    ):
        super().__init__()
        self.rank_approximation = rank_approximation
        self.return_type = return_type
        assert return_type in ["1-mAP", "1-AP", "AP", 'mAP']

    def general_forward(self, scores, target, verbose=False):
        batch_size = target.size(0)
        nb_instances = target.size(1)
        device = scores.device

        ap_score = []
        mask = (1 - torch.eye(nb_instances, device=device))
        iterator = range(batch_size)
        if verbose:
            iterator = tqdm(iterator, leave=None)
        for idx in iterator:
            # shape M
            query = scores[idx]
            pos_mask = target[idx].bool()

            # shape M x M
            query = query.view(1, -1) - query[pos_mask].view(-1, 1)
            query = self.rank_approximation(query, target=pos_mask, general=True) * mask[pos_mask]

            # shape M
            rk = 1 + query.sum(-1)

            # shape M
            pos_rk = 1 + (query * pos_mask.view(1, -1)).sum(-1)

            # shape 1
            ap = (pos_rk / rk).sum(-1) / pos_mask.sum()
            ap_score.append(ap)

        # shape N
        ap_score = torch.stack(ap_score)

        return ap_score

    def quick_forward(self, scores, target):
        batch_size = target.size(0)
        device = scores.device

        # ------ differentiable ranking of all retrieval set ------
        # compute the mask which ignores the relevance score of the query to itself
        mask = 1.0 - torch.eye(batch_size, device=device).unsqueeze(0)
        # compute the relevance scores via cosine similarity of the CNN-produced embedding vectors
        # compute the difference matrix
        sim_diff = scores.unsqueeze(1) - scores.unsqueeze(1).permute(0, 2, 1)

        # pass through the sigmoid
        sim_diff_sigmoid = self.rank_approximation(sim_diff, target=target)

        sim_sg = sim_diff_sigmoid * mask
        # compute the rankings
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1

        # ------ differentiable ranking of only positive set in retrieval set ------
        # compute the mask which only gives non-zero weights to the positive set
        pos_mask = (target - torch.eye(batch_size).to(device))
        sim_pos_sg = sim_diff_sigmoid * pos_mask
        sim_pos_rk = (torch.sum(sim_pos_sg, dim=-1) + target) * target
        # compute the rankings of the positive set

        ap = ((sim_pos_rk / sim_all_rk).sum(1) * (1 / target.sum(1)))
        return ap

    def forward(self, scores, target, force_general=False, verbose=False):
        assert scores.shape == target.shape
        assert len(scores.shape) == 2

        if (scores.size(0) == scores.size(1)) and not force_general:
            ap = self.quick_forward(scores, target)
        else:
            ap = self.general_forward(scores, target, verbose=verbose)

        if self.return_type == 'AP':
            return ap
        elif self.return_type == 'mAP':
            return ap.mean()
        elif self.return_type == '1-AP':
            return 1 - ap
        elif self.return_type == '1-mAP':
            return 1 - ap.mean()

    @property
    def my_repr(self,):
        repr = f"return_type={self.return_type}"
        return repr


class SmoothAP(SmoothRankAP):

    def __init__(self, tau=0.01, **kwargs):
        rank_approximation = partial(tau_sigmoid, tau=tau)
        super().__init__(rank_approximation, **kwargs)
        self.tau = tau

    def extra_repr(self,):
        repr = f"tau={self.tau}, {self.my_repr}"
        return repr



class SoftBinAP (nn.Module):
    """ Differentiable AP loss, through quantization. From the paper:
        Learning with Average Precision: Training Image Retrieval with a Listwise Loss
        Jerome Revaud, Jon Almazan, Rafael Sampaio de Rezende, Cesar de Souza
        https://arxiv.org/abs/1906.07589
        Input: (N, M)   values in [min, max]
        label: (N, M)   values in {0, 1}
        Returns: 1 - mAP (mean AP for each n in {1..N})
                 Note: typically, this is what you wanna minimize
    """
    def __init__(
        self,
        nq=10,
        min=0,
        max=1,
        return_type='1-mAP',
    ):
        super().__init__()
        assert isinstance(nq, int) and 2 <= nq <= 100
        assert return_type in ["1-mAP", "AP", "1-AP", "mAP", "debug"]
        self.nq = nq
        self.min = min
        self.max = max
        self.return_type = return_type

        gap = max - min
        assert gap > 0
        # Initialize quantizer as non-trainable convolution
        self.quantizer = q = nn.Conv1d(1, 2*nq, kernel_size=1, bias=True)
        q.weight = nn.Parameter(q.weight.detach(), requires_grad=False)
        q.bias = nn.Parameter(q.bias.detach(), requires_grad=False)
        a = (nq-1) / gap
        # First half equal to lines passing to (min+x,1) and (min+x+1/a,0)
        # with x = {nq-1..0}*gap/(nq-1)
        q.weight[:nq] = -a
        q.bias[:nq] = torch.from_numpy(a*min + np.arange(nq, 0, -1))  # b = 1 + a*(min+x)
        # First half equal to lines passing to (min+x,1) and (min+x-1/a,0)
        # with x = {nq-1..0}*gap/(nq-1)
        q.weight[nq:] = a
        q.bias[nq:] = torch.from_numpy(np.arange(2-nq, 2, 1) - a*min)  # b = 1 - a*(min+x)
        # First and last one as a horizontal straight line
        q.weight[0] = q.weight[-1] = 0
        q.bias[0] = q.bias[-1] = 1

    def forward(self, x, label, qw=None):
        assert x.shape == label.shape  # N x M
        N, M = x.shape
        # Quantize all predictions
        q = self.quantizer(x.unsqueeze(1))
        q = torch.min(q[:, :self.nq], q[:, self.nq:]).clamp(min=0)  # N x Q x M

        nbs = q.sum(dim=-1)  # number of samples  N x Q = c
        rec = (q * label.view(N, 1, M).float()).sum(dim=-1)  # number of correct samples = c+ N x Q
        prec = rec.cumsum(dim=-1) / (1e-16 + nbs.cumsum(dim=-1))  # precision
        rec /= rec.sum(dim=-1).unsqueeze(1)  # norm in [0,1]

        ap = (prec * rec).sum(dim=-1)  # per-image AP

        if self.return_type == '1-mAP':
            if qw is not None:
                ap *= qw  # query weights
            loss = 1 - ap.mean()
            return loss
        elif self.return_type == 'AP':
            assert qw is None
            return ap
        elif self.return_type == 'mAP':
            assert qw is None
            return ap.mean()
        elif self.return_type == '1-AP':
            return 1 - ap
        elif self.return_type == 'debug':
            return prec, rec

    def extra_repr(self,):
        return f"nq={self.nq}, min={self.min}, max={self.max}, return_type={self.return_type}"


class SupAP(SmoothRankAP):

    def __init__(self, tau=0.01, rho=100, offset=None, delta=0.05, start=0.5, **kwargs):
        rank_approximation = partial(step_rank, tau=tau, rho=rho, offset=offset, delta=delta, start=start)
        super().__init__(rank_approximation, **kwargs)
        self.tau = tau
        self.rho = rho
        self.offset = offset
        self.delta = delta

    def extra_repr(self,):
        repr = f"tau={self.tau}, rho={self.rho}, offset={self.offset}, delta={self.delta}, {self.my_repr}"
        return repr


def heaviside(tens, val=1., target=None, general=None):
    return (tens > 0).float() * 1 + (tens < 0).float() * 0 + (tens == 0).float() * torch.tensor(val, device=tens.device, dtype=tens.dtype)
    return torch.heaviside(tens, values=torch.tensor(val, device=tens.device, dtype=tens.dtype))




def step_rank(tens, tau, rho, offset=None, delta=None, start=0.5, target=None, general=False):
    target = target.squeeze()
    if general:
        target = target.view(1, -1).repeat(tens.size(0), 1)
    else:
        mask = target.unsqueeze(target.ndim - 1).bool()
        target = lib.create_label_matrix(target).bool() * mask
    pos_mask = (tens > 0).bool()
    neg_mask = ~pos_mask

    if isinstance(tau, str):
        tau_n, tau_p = tau.split("_")
    else:
        tau_n = tau_p = tau

    if delta is None:
        tens[~target & pos_mask] = rho * tens[~target & pos_mask] + offset
    else:
        margin_mask = tens > delta
        tens[~target & pos_mask & ~margin_mask] = start + tau_sigmoid(tens[~target & pos_mask & ~margin_mask], tau_p).type(tens.dtype)
        if offset is None: 
            offset = tau_sigmoid(torch.tensor([delta], device=tens.device), tau_p).type(tens.dtype) + start
        tens[~target & pos_mask & margin_mask] = rho * (tens[~target & pos_mask & margin_mask] - delta) + offset

    tens[~target & neg_mask] = tau_sigmoid(tens[~target & neg_mask], tau_n).type(tens.dtype)

    tens[target] = heaviside(tens[target], val=torch.tensor(1., device=tens.device, dtype=tens.dtype))

    return tens
