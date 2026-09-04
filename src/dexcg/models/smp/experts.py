"""Independent diffusion experts for scalar skill coefficients."""

import math
from collections.abc import Sequence

import torch
from torch import nn


class SinusoidalTimestep(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / (half - 1)
        frequencies = torch.exp(
            torch.arange(half, device=timestep.device, dtype=torch.float32) * -scale
        )
        angles = timestep.float()[:, None] * frequencies[None]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, groups: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(groups, output_dim),
            nn.Mish(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ConditionalBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        condition_dim: int,
        kernel_size: int,
        groups: int,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.first = ConvBlock(input_dim, output_dim, kernel_size, groups)
        self.second = ConvBlock(output_dim, output_dim, kernel_size, groups)
        self.condition = nn.Sequential(nn.Mish(), nn.Linear(condition_dim, output_dim * 2))
        self.residual = (
            nn.Conv1d(input_dim, output_dim, 1) if input_dim != output_dim else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        outputs = self.first(inputs)
        scale, bias = self.condition(condition).chunk(2, dim=-1)
        outputs = scale.unsqueeze(-1) * outputs + bias.unsqueeze(-1)
        return self.second(outputs) + self.residual(inputs)


class CoefficientUNet(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        timestep_dim: int = 128,
        down_dims: Sequence[int] = (64, 128, 256),
        kernel_size: int = 3,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.timestep_encoder = nn.Sequential(
            SinusoidalTimestep(timestep_dim),
            nn.Linear(timestep_dim, timestep_dim * 4),
            nn.Mish(),
            nn.Linear(timestep_dim * 4, timestep_dim),
        )
        full_condition_dim = timestep_dim + condition_dim
        dimensions = (1, *down_dims)
        pairs = list(zip(dimensions[:-1], dimensions[1:]))
        self.down = nn.ModuleList()
        for index, (input_dim, output_dim) in enumerate(pairs):
            self.down.append(
                nn.ModuleList(
                    (
                        ConditionalBlock(
                            input_dim, output_dim, full_condition_dim, kernel_size, groups
                        ),
                        ConditionalBlock(
                            output_dim, output_dim, full_condition_dim, kernel_size, groups
                        ),
                        nn.Conv1d(output_dim, output_dim, 3, 2, 1)
                        if index < len(pairs) - 1
                        else nn.Identity(),
                    )
                )
            )
        self.middle = nn.ModuleList(
            ConditionalBlock(down_dims[-1], down_dims[-1], full_condition_dim, kernel_size, groups)
            for _ in range(2)
        )
        self.up = nn.ModuleList()
        for input_dim, output_dim in reversed(pairs[1:]):
            self.up.append(
                nn.ModuleList(
                    (
                        ConditionalBlock(
                            output_dim * 2, input_dim, full_condition_dim, kernel_size, groups
                        ),
                        ConditionalBlock(
                            input_dim, input_dim, full_condition_dim, kernel_size, groups
                        ),
                        nn.ConvTranspose1d(input_dim, input_dim, 4, 2, 1),
                    )
                )
            )
        self.output = nn.Sequential(
            ConvBlock(down_dims[0], down_dims[0], kernel_size, groups),
            nn.Conv1d(down_dims[0], 1, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        values = sample.transpose(1, 2)
        timestep = torch.as_tensor(timestep, device=values.device).reshape(-1)
        timestep = timestep.expand(values.shape[0])
        global_condition = torch.cat((self.timestep_encoder(timestep), condition), dim=-1)
        skips = []
        for first, second, downsample in self.down:
            values = second(first(values, global_condition), global_condition)
            skips.append(values)
            values = downsample(values)
        for block in self.middle:
            values = block(values, global_condition)
        for first, second, upsample in self.up:
            values = torch.cat((values, skips.pop()), dim=1)
            values = upsample(second(first(values, global_condition), global_condition))
        return self.output(values).transpose(1, 2)


class CoefficientExperts(nn.Module):
    def __init__(
        self,
        num_experts: int,
        condition_dim: int,
        timestep_dim: int = 128,
        down_dims: Sequence[int] = (64, 128, 256),
        kernel_size: int = 3,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            CoefficientUNet(condition_dim, timestep_dim, down_dims, kernel_size, groups)
            for _ in range(num_experts)
        )

    def forward(
        self,
        noisy_coefficients: torch.Tensor,
        timestep: torch.Tensor | int,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        predictions = [
            expert(noisy_coefficients[..., index : index + 1], timestep, condition)
            for index, expert in enumerate(self.experts)
        ]
        return torch.cat(predictions, dim=-1)
