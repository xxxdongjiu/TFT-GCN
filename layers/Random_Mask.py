import torch


def random_masking_3D(xb, mask_ratio):
    # xb: [bs x dim x L x patch]
    bs, C, L, p = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(bs, C, L, device=xb.device)  # noise in [0, 1], bs x C x L

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=2)  # ascend: small is keep, large is removed
    ids_restore = torch.argsort(ids_shuffle, dim=2)  # ids_restore: [bs x C x L]

    # keep the first subset
    ids_keep = ids_shuffle[:, :, :len_keep]  # ids_keep: [bs x C x len_keep]
    x_kept = torch.gather(
        x, dim=2, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, p)
    )  # x_kept: [bs x C x len_keep x patch]

    # removed x
    x_removed = torch.zeros(bs, C, L - len_keep, p, device=xb.device)  # x_removed: [bs x C x (L-len_keep) x patch]
    x_ = torch.cat([x_kept, x_removed], dim=2)  # x_: [bs x C x L x patch]

    # combine the kept part and the removed one
    x_masked = torch.gather(
        x_, dim=2, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, p)
    )  # x_masked: [bs x C x L x patch]
    x_masked = x_masked.reshape(bs, C, -1).transpose(1, 2)

    # generate the binary mask: 0 is keep, 1 is removed
    mask = torch.ones([bs, C, L, p], device=xb.device)  # mask: [bs x C x L x patch]
    mask[:, :, :len_keep, :] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=2, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1, p))  # [bs x C x L x patch]
    mask = mask.reshape(bs, C, -1).transpose(1, 2)
    return x_masked, mask
