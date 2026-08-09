from typing import List, Dict
import torch
import torchvision


def crop(rendered_views: torch.Tensor) -> torch.Tensor:
    """Crop the renderedviews"""

    w = rendered_views.size(2)
    transforms = torchvision.transforms.Resize((w, w))
    for i in range(rendered_views.size(0)):
        mask = rendered_views[i][:, :, 3].squeeze()
        non_zero_indices = mask.nonzero()

        # # Get the minimum and maximum indices along each dimension
        # min_h, max_h = non_zero_indices[:, 0].min(), non_zero_indices[:, 0].max()
        # min_w, max_w = non_zero_indices[:, 1].min(), non_zero_indices[:, 1].max()

        # Get minimum and maximum corner of the non-empty region
        # domain: [0,pixel_count], pixel centers are at coordinates x+0.5
        bb_min = non_zero_indices.min(dim=0)[0]
        bb_max = non_zero_indices.max(dim=0)[0] + 1
        bb_center = 0.5 * (bb_min + bb_max)

        # Get maxmimum height and width the crop can have without going out of bounds given the center
        img_size = torch.tensor(mask.shape, device=bb_center.device, dtype=torch.float32)
        max_hw = torch.minimum(2 * (img_size - bb_center), 2 * bb_center)

        # Get size of the crop region
        size = (bb_max - bb_min).max() * 1.1
        size = torch.clamp(size, max=torch.min(max_hw))

        # Get min and max pixel indices of the crop region (max is past-the-end)
        crop_min = torch.ceil(-0.5 + bb_center - 0.5 * size).to(torch.int64)
        crop_max = crop_min + torch.round(size).to(torch.int64)
        min_h = crop_min[0].item()
        min_w = crop_min[1].item()
        max_h = crop_max[0].item()
        max_w = crop_max[1].item()
        size = max_w - min_w

        # # Calculate the size of the square region
        # size = max(max_h - min_h + 1, max_w - min_w + 1) * 1.1
        # size = torch.clamp(size, max=min(maps["mask"].shape))
        # max_w - min_w

        # # Calculate the upper left corner of the square region
        # h_start = min(min_h, max_h) - (size - (max_h - min_h + 1)) / 2
        # w_start = min(min_w, max_w) - (size - (max_w - min_w + 1)) / 2

        # min_h = int(h_start)
        # min_w = int(w_start)
        # max_h = int(min_h + size)
        # max_w = int(min_w + size)
        # size = max_w - min_w

        rendered_views[i] = transforms(rendered_views[i][min_h:max_h, min_w:max_w].permute(2, 0, 1)).permute(1, 2, 0)

    return rendered_views


def crop_composited(
    composited: torch.Tensor, alpha: torch.Tensor, expand: float = 1.1
) -> torch.Tensor:
    """Tight square crop of each composited view around its figure, resized back.

    The spherical port of the planar ``crop`` above (GEM's CROP_RENDERINGS): SDS
    always sees "one figure, framed tight, at maximum pixel scale" regardless of
    how the outline deforms -- while the CAMERA keeps its margin, which is what
    lets ``dr.antialias`` produce outline gradients at all. Out-of-place (the
    planar version mutates its input and throws on empty alpha, which the caller
    papers over with a bare ``except``); empty alpha passes through untouched.
    The crop window indices are non-differentiable, but the crop is applied after
    antialias+compositing, so gradients flow through the slice and resize.

    Args:
        composited: ``(B, H, W, 3)`` background-composited renders.
        alpha: ``(B, H, W, 1)`` coverage from the same render.
    """
    import torch.nn.functional as F

    b, h, w, _ = composited.shape
    out = []
    for i in range(b):
        mask = alpha[i, ..., 0] > 0
        if not mask.any():
            out.append(composited[i])
            continue
        rows = torch.where(mask.any(dim=1))[0]
        cols = torch.where(mask.any(dim=0))[0]
        r0, r1 = int(rows[0]), int(rows[-1])
        c0, c1 = int(cols[0]), int(cols[-1])
        side = min(int(max(r1 - r0 + 1, c1 - c0 + 1) * expand), h, w)
        top = max(0, min((r0 + r1 + 1) // 2 - side // 2, h - side))
        left = max(0, min((c0 + c1 + 1) // 2 - side // 2, w - side))
        patch = composited[i, top : top + side, left : left + side, :]
        patch = F.interpolate(
            patch.permute(2, 0, 1).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0)
        out.append(patch)
    return torch.stack(out)
