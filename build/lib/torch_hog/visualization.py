from functools import lru_cache
import torch
import math


class HOGVisualization:
    @lru_cache(maxsize=1)
    def _generateAngleLines(self, phaseMin=10, phaseMax=170):
        """
        Generates angle lines for visualization.

        Args:
            phaseMin (int, optional): The minimum phase value. Defaults to 10.
            phaseMax (int, optional): The maximum phase value. Defaults to 170.
            cellSize (int, optional): The size of each cell in pixels. 
            Defaults to 16.

        Returns:
            torch.Tensor: The generated angle lines.
        """
        angle = (torch.linspace(phaseMin, phaseMax, 9)) / 180 * math.pi
        zeroPhaseLine = torch.zeros(self.cellSize, self.cellSize)
        centerCellSize = self.cellSize // 2
        zeroPhaseLine[centerCellSize] = 1
        coords = torch.dstack(
            torch.meshgrid(
                *2 * [torch.linspace(-1, 1, self.cellSize).to(torch.float32)], 
                indexing="xy"
            )
        )
        rotMat = torch.stack(
            (
                torch.cos(angle), -torch.sin(angle),
                torch.sin(angle), torch.cos(angle)
            ), dim=-1
        ).view(-1, 2, 2)
        return (
            torch.nn.functional.grid_sample(
                zeroPhaseLine.expand(angle.size(0), 1, -1, -1),
                torch.einsum("ijk,bkl->bijl", coords, rotMat),
                align_corners=False,
                mode="bilinear",
            )
            .moveaxis(0, -1)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )
