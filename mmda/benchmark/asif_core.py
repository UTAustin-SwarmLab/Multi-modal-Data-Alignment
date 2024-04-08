"""Code modified from the original implementation by the authors of the ASIF paper.

Paper title: ASIF: Coupled Data Turns Unimodal Models to Multimodal Without Training.
Github:
https://colab.research.google.com/github/noranta4/ASIF/blob/main/ASIF_colab_demo.ipynb#scrollTo=x51O0Ndmj1Sy
"""

import torch


def relative_represent(
    y: torch.Tensor, basis: torch.Tensor, non_zeros: int = 800, max_gpu_mem_gb: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the sparse decomposition of a tensor y with respect to a basis, considering the available GPU memory.

    Args:
        y (torch.Tensor): Vectors to represent.
        basis (torch.Tensor): Basis to represent with respect to.
        non_zeros (int): Nonzero entries in the relative representation.
        max_gpu_mem_gb (int): Maximum GPU memory allowed to use in gigabytes.

    Returns:
        indices (torch.Tensor): Indices of the nonzero entries in each relative representation of y.
        values (torch.Tensor): Corresponding coefficients of the entries.
    """
    values, indices = torch.zeros((y.shape[0], non_zeros)), torch.zeros((y.shape[0], non_zeros), dtype=torch.long)

    free_gpu_mem = max_gpu_mem_gb * 1024**3
    max_floats_in_mem = free_gpu_mem / 4
    max_chunk_y = max_floats_in_mem / basis.shape[0]
    n_chunks = int(y.shape[0] / max_chunk_y) + 1
    chunk_y = int(y.shape[0] / n_chunks) + n_chunks

    with torch.no_grad():
        for c in range(n_chunks):
            in_prods = torch.einsum("ik, jk -> ij", y[c * chunk_y : (c + 1) * chunk_y], basis)
            values[c * chunk_y : (c + 1) * chunk_y], indices[c * chunk_y : (c + 1) * chunk_y] = torch.topk(
                in_prods, non_zeros, dim=1
            )
            del in_prods

    return indices.to("cpu"), values.to("cpu")


def sparsify(i: torch.Tensor, v: torch.Tensor, size: torch.Size) -> torch.sparse.FloatTensor:
    """Organize indices and values of n vectors into a single sparse tensor.

    Args:
        i (torch.Tensor): indices of non-zero elements of every vector. Shape: (n_vectors, nonzero elements)
        v (torch.Tensor): values of non-zero elements of every vector. Shape: (n_vectors, nonzero elements)
        size (torch.Size): shape of the output tensor

    Returns:
        torch.sparse.FloatTensor: sparse tensor of shape "size" (n_vectors, zero + nonzero elements)
    """
    flat_dim = len(i.flatten())
    coo_first_row_idxs = torch.div(torch.arange(flat_dim), i.shape[1], rounding_mode="floor")
    stacked_idxs = torch.cat((coo_first_row_idxs.unsqueeze(0), i.flatten().unsqueeze(0)), 0)
    return torch.sparse_coo_tensor(stacked_idxs, v.flatten(), size)


def normalize_sparse(tensor: torch.sparse.FloatTensor, nnz_per_row: int) -> torch.sparse.FloatTensor:
    """Normalize a sparse tensor by row.

    Args:
        tensor (torch.sparse.FloatTensor): The sparse tensor to normalize.
        nnz_per_row (int): The number of non-zero elements per row.

    Returns:
        torch.sparse.FloatTensor: The normalized sparse tensor.
    """
    norms = torch.sparse.sum(tensor * tensor, dim=1).to_dense()
    v = tensor._values().clone().detach().reshape(-1, nnz_per_row).t()
    v /= torch.sqrt(norms)
    return torch.sparse_coo_tensor(tensor._indices(), v.t().flatten(), tensor.shape)


def zero_shot_classification(
    zimgs: torch.Tensor,
    ztxts: torch.Tensor,
    aimgs: torch.Tensor,
    atxts: torch.Tensor,
    test_labels: list,
    non_zeros: int,
    range_anch: range,
    val_exps: list,
    dic_size: int = 100_000,
    max_gpu_mem_gb: float = 8.0,
) -> tuple[list, dict, torch.Tensor]:
    """Computes the zero-shot classification accuracy using relative representations over sets of anchors.

    Args:
        zimgs (torch.Tensor): absolute embeddings of the images
        ztxts (torch.Tensor): absolute embeddings of the texts
        aimgs (torch.Tensor): absolute embeddings of the anchor images
        atxts (torch.Tensor): absolute embeddings of the anchor texts
        test_labels (list): ground truth labels of the images
        non_zeros (int): nonzero entries in the relative representation
        range_anch (range): range of sizes of the anchor's sets to use (overshoot is ok)
        val_exps (list): similarity exponents to test
        dic_size (int): size of the chunk of aimgs to load in memory to fit all intermediate variables in RAM
        max_gpu_mem_gb (float): maximum GPU memory allowed to use in gigabytes
    Returns:
        n_anchors (list): list of sizes of the anchor's sets (with overshooting fixed)
        scores (dict): dictionary of scores for each tested similarity exponent
        sims (torch.Tensor): similarity matrix between images and texts
    """
    n_anchors = []
    scores = {ve: [] for ve in val_exps}
    n_templates = max(int(ztxts.shape[0] / (max(test_labels) - min(test_labels) + 1)), 1)

    for i in range_anch:
        sims = torch.zeros((len(zimgs), len(ztxts)))
        idxs_imgs = torch.zeros(((len(zimgs), non_zeros * 2)), dtype=torch.long)
        idxs_txts = torch.zeros(((len(ztxts), non_zeros * 2)), dtype=torch.long)
        vals_imgs = torch.zeros((len(zimgs), non_zeros * 2))
        vals_txts = torch.zeros((len(ztxts), non_zeros * 2))

        for d in range(min(len(aimgs), i) // (dic_size + 1) + 1):
            idxs, values = relative_represent(
                zimgs,
                aimgs[d * dic_size : min(i, (d + 1) * dic_size)],
                non_zeros=non_zeros,
                max_gpu_mem_gb=max_gpu_mem_gb,
            )
            idxs_imgs[:, :non_zeros] = idxs + d * dic_size
            vals_imgs[:, :non_zeros] = values
            idxs, values = relative_represent(
                ztxts,
                atxts[d * dic_size : min(i, (d + 1) * dic_size)],
                non_zeros=non_zeros,
                max_gpu_mem_gb=max_gpu_mem_gb,
            )
            idxs_txts[:, :non_zeros] = idxs + d * dic_size
            vals_txts[:, :non_zeros] = values

            top_valsi, indices = torch.topk(vals_imgs, non_zeros, dim=1)
            top_idxsi = torch.gather(idxs_imgs, 1, indices)
            top_valst, indices = torch.topk(vals_txts, non_zeros, dim=1)
            top_idxst = torch.gather(idxs_txts, 1, indices)

            idxs_imgs[:, non_zeros:] = top_idxsi
            vals_imgs[:, non_zeros:] = top_valsi
            idxs_txts[:, non_zeros:] = top_idxst
            vals_txts[:, non_zeros:] = top_valst

        for val_exp in val_exps:
            ztxts_t = sparsify(top_idxst, top_valst**val_exp, (len(ztxts), min(len(aimgs), i))).to(zimgs.device)
            ztxts_t = normalize_sparse(ztxts_t, non_zeros)

            if i < max_gpu_mem_gb * 1024**3 / 4 / zimgs.shape[0]:  # einsum until it fits in GPU memory
                zimgs_t = sparsify(top_idxsi, top_valsi**val_exp, (len(zimgs), min(len(aimgs), i))).to(zimgs.device)
                sims = torch.einsum("ij, kj -> ik", zimgs_t.to_dense(), ztxts_t.to_dense()).to("cpu")
            else:
                n_chunks = 6
                zs = zimgs.shape[0]
                chunks = [c * (zs // n_chunks) for c in range(n_chunks)] + [zs]
                for ci in range(n_chunks):
                    zimgs_t = sparsify(
                        top_idxsi[chunks[ci] : chunks[ci + 1]],
                        top_valsi[chunks[ci] : chunks[ci + 1]] ** val_exp,
                        (chunks[ci + 1] - chunks[ci], min(len(aimgs), i)),
                    ).to(zimgs.device)
                    sims[chunks[ci] : chunks[ci + 1]] = torch.sparse.mm(zimgs_t, ztxts_t.t()).to("cpu").to_dense()
            score = float(
                (
                    torch.div(sims.argmax(axis=1), n_templates, rounding_mode="floor") == test_labels.clone().detach()
                ).sum()
                # (torch.div(sims.argmax(axis=1), n_templates, rounding_mode="floor") == torch.tensor(test_labels)).sum()
                / len(zimgs)
            )
            scores[val_exp].append(score)
        n_anchors.append(min(len(aimgs), i))
    return n_anchors, scores, sims
