from typing import Literal
import torch
import math
from .parametrization import HOGParametrization
from .visualization import HOGVisualization
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


class HOG(torch.nn.Module, HOGParametrization, HOGVisualization):
    """
    HOG (Histogram of Oriented Gradients) module.

    Args:
        cellSize (int): Size of the cell in pixels. Default is 16.
        blockSize (int): Size of the block in cells. Default is 2.
        nPhaseBins (int): Number of phase bins. Default is 9.
        kernel (Literal["sobel", "finite"]): Type of gradient kernel.
            Default is "finite".
        normalization (Literal["L2", "L1", "L2Hys"]): Type of normalization.
            Default is "L2".
        accumulate (Literal["simple", "bilinear"]): Type of accumulation.
            Default is "bilinear".
        channelWise (bool): Whether to perform channel-wise reduction.
            Default is False.
    """

    def __init__(
        self,
        cellSize: int = 16,
        blockSize: int = 2,
        nPhaseBins: int = 9,
        kernel: Literal["sobel", "finite"] = "finite",
        normalization: Literal["L2", "L1", "L2Hys"] = "L2",
        accumulate: Literal["simple", "bilinear"] = "bilinear",
        channelWise: bool = False,
    ):
        super(HOG, self).__init__()
        # Initialize parameters
        if not isinstance(cellSize, int) or cellSize < 1:
            raise ValueError("cellSize must be a positive integer")
        self.cellSize = cellSize
        if not isinstance(blockSize, int) or blockSize < 1:
            raise ValueError("blockSize must be a positive integer")
        self.blockSize = blockSize
        if not isinstance(nPhaseBins, int) or nPhaseBins < 1:
            raise ValueError("nPhaseBins must be an integer >1")
        self.nPhaseBins = nPhaseBins
        self.phaseGain = self.nPhaseBins / math.pi
        if not isinstance(accumulate, str) or accumulate not in [
            "simple", "bilinear"
        ]:
            raise ValueError(
                "accumulate must be either 'simple' or 'bilinear'"
            )
        self.accumulate = self._chooseAccumulateFunc(accumulate)
        if not isinstance(channelWise, bool):
            raise ValueError("channelWise must be a boolean")
        self.channelWise = self._chooseChannelsReduceFunc(channelWise)
        self.grad = self._chooseGradFunc(kernel)
        self.normalization = self._chooseNormalizationFunc(normalization)

    def forward(self, im: torch.Tensor):
        """
        Forward pass of the HOG module.

        Args:
            im (torch.Tensor): Input image tensor (B,C,H,W).

        Returns:
            torch.Tensor: Normalized HOG features
                (H//cellSize, W//cellSize, blockSize, blockSize, nPhaseBins)
        """
        # Compute cells histograms, unfold to get blocks and then normalize 
        # per block
        return self.blockNormalize(self.hog(im))

    def hog(self, im: torch.Tensor):
        """
        Compute HOG features for the input image.

        Args:
            im (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: HOG features.
        """
        # Compute gradients and then unfold to get cells
        grads = (
            self.grad(im)
            .unfold(-2, *2 * [self.cellSize])
            .unfold(-2, *2 * [self.cellSize])
            .flatten(-2, -1)
        )
        # Compute norm and reduce channel dimension if asked
        norm, grads = self.channelWise(grads.abs(), grads)
        return self.cellsHist(norm, grads.angle() % math.pi)

    def cellsHist(self, norm, phase):
        """
        Compute histograms of cells.

        Args:
            norm (torch.Tensor): Norm of gradients.
            phase (torch.Tensor): Phase of gradients.

        Returns:
            torch.Tensor: Histograms of cells.
        """
        binsEdges = torch.linspace(0, math.pi, self.nPhaseBins + 1)
        leftEdgeIndices = (self.phaseGain * phase).floor().to(torch.int64)
        return (
            self.accumulate(
                torch.zeros((*phase.shape[:-1], self.nPhaseBins)),
                leftEdgeIndices,
                norm,
                phase,
                binsEdges,
            )
            / self.cellSize**2
        )

    def blockNormalize(self, hog: torch.Tensor):
        """
        Normalize HOG features per block.

        Args:
            hog (torch.Tensor): HOG features.

        Returns:
            torch.Tensor: Normalized HOG features.
        """
        return self.normalization(
            hog.unfold(
                -3, self.blockSize, halfBlockSize := self.blockSize // 2
            )
            .unfold(-3, self.blockSize, halfBlockSize)
            .moveaxis(-3, -1)
        )

    def visualize(self, im: torch.Tensor, orthogonal=False):
        """
        Visualize HOG features on the input image.

        Args:
            im (torch.Tensor): Input image tensor.
            orthogonal (bool): Whether to visualize orthogonal lines. 
                Default is False.

        Returns:
            torch.Tensor: Visualized HOG features.
        """
        lines = self._generateAngleLines(
            phaseMin=10 + 1 * orthogonal * 90,
            phaseMax=170 + 1 * orthogonal * 90
        )
        hist = self.hog(im)
        return (
            (lines * (hist).unsqueeze(-2).unsqueeze(-2))
            .sum(dim=-1)
            .moveaxis(-2, -3)
            .flatten(-4, -3)
            .flatten(-2, -1)
        )

    def plotGrads(self, im, batch=0, figsize=(12, 3)):
        """
        Plot the gradients of the input image.

        Args:
            im (torch.Tensor): Input image tensor.
            batch (int): Batch index. Default is 0.
            figsize (tuple): Figure size. Default is (12, 3).
        """
        def normalize(im):
            minV, maxV = torch.aminmax(im)
            return (im - minV) / (maxV - minV)

        hist = self.hog(im).moveaxis(-1, 0)
        _, axes = plt.subplots(1, hist.size(0), figsize=figsize)
        for i, (ax, binIm) in enumerate(zip(axes, hist)):
            ax.imshow(
                normalize(binIm[batch]).moveaxis(0, -1),
                cmap="gray",
            )
            ax.set_axis_off()
            ax.set_title(f"{i+1}th")
        plt.show()

    def plotVisualize(self, im, orthogonal=False, batch=0):
        def normalize(im):
            minV, maxV = torch.aminmax(im)
            return (im - minV) / (maxV - minV)
        
        cmapsRGB = [
            LinearSegmentedColormap.from_list(
                color,
                [oh := [1 if i == k else 0 for i in range(3)], oh],
            )
            for k, color in enumerate(["R", "G", "B"])
        ]

        vis = self.visualize(im, orthogonal=orthogonal)
        plt.imshow(normalize(im[batch]).moveaxis(0, -1))
        for ch in range(vis.size(1)):
            plt.imshow(
                normed := normalize(vis[batch, ch]),
                cmap=cmapsRGB[ch] if vis.size(1) > 1 else "gray",
                alpha=normed.clamp(max=0.3) / 0.3,
            )
        plt.show()
    