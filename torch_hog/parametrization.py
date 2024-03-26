
from typing import Literal
import torch
import torch.nn.functional as F


class HOGParametrization:
    SOBEL_KERNEL = torch.tensor(
        [
            [-1.0 - 1.0j, 0.0 - 2.0j, 1.0 - 1.0j],
            [-2.0 + 0.0j, 0.0 + 0.0j, 2.0 + 0.0j],
            [-1.0 + 1.0j, 0.0 + 2.0j, 1.0 + 1.0j],
        ],
        dtype=torch.complex64,
    )

    def _chooseChannelsReduceFunc(self, channelWise: bool):
        def channelsMax(norm, grads):
            return (
                torch.gather(
                    norm,
                    1,
                    normAMaxChan := norm.argmax(dim=1, keepdim=True)
                ),
                torch.gather(grads, 1, normAMaxChan),
            )

        def channelsIdentity(norm, grads):
            return norm, grads

        if channelWise:
            return channelsIdentity
        else:
            return channelsMax

    def _chooseAccumulateFunc(self, accumulate: Literal["simple", "bilinear"]):
        def accumulateSimple(zeros, leftEdgeIndices, norm, *_):
            return zeros.scatter_add(
                -1,
                leftEdgeIndices % self.nPhaseBins,
                norm
            )

        def accumulateBilinear(zeros, leftEdgeIndices, norm, phase, binsEdges):
            leftEdgePrct = self.phaseGain * (
                phase - binsEdges[leftEdgeIndices]
            )  # Compute left edge distance percentage
            return zeros.scatter_add(
                -1,
                leftEdgeIndices % self.nPhaseBins,
                (1 - leftEdgePrct) * norm,
            ).scatter_add(  # Add weighted norm to left edge
                -1, (leftEdgeIndices + 1) % self.nPhaseBins,
                leftEdgePrct * norm
            )  # Add weighted norm to right edge

        match accumulate:
            case "simple":
                return accumulateSimple
            case "bilinear":
                return accumulateBilinear
            case _:
                raise ValueError("kernel must be either 'sobel' or 'finite'")

    def _chooseGradFunc(self, kernel: Literal["sobel", "finite"]):
        def gradFinite(im):
            # This method a bit faster than F.conv2d
            # especially for late computing the abs and angle
            grads = torch.zeros_like(im, dtype=torch.complex64)
            grads[:, :, :, 1:-1] = im[:, :, :, 2:] - im[:, :, :, :-2]
            grads[:, :, 1:-1, :] += 1j * (im[:, :, 2:, :] - im[:, :, :-2, :])
            return grads

        def gradSobel(im):
            kernel = HOGParametrization.SOBEL_KERNEL.expand(
                im.size(1), 1, -1, -1
            )
            return F.conv2d(
                torch.nn.functional.pad(
                    im.to(kernel.dtype), 4 * [kernel.shape[0] // 2],
                    mode="reflect"
                ),
                kernel,
                groups=im.size(1),
            )

        match kernel:
            case "finite":
                return gradFinite
            case "sobel":
                return gradSobel
            case _:
                raise ValueError("kernel must be either 'sobel' or 'finite'")

    def _chooseNormalizationFunc(self, norm: Literal["L1", "L2"]):
        def normL2(x):
            return x / (x.norm(p=2, dim=(-1, -2, -3), keepdim=True) + 1e-10)

        def normL1(x):
            return x / (x.norm(p=1, dim=(-1, -2, -3), keepdim=True) + 1e-10)

        def normL2Hys(x):
            return (
                trhold := (
                    x / (x.norm(p=2, dim=(-1, -2, -3), keepdim=True) + 1e-10)
                ).clamp(max=0.2)
            ) / (trhold.norm(p=2, dim=(-1, -2, -3), keepdim=True) + 1e-10)

        match norm:
            case "L1":
                return normL1
            case "L2":
                return normL2
            case "L2Hys":
                return normL2Hys
            case _:
                raise ValueError("normalization:norm must be either 'L1'\
or 'L2'")


