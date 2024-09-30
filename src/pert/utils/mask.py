import torch

from math import ceil
from abc import abstractmethod
from typing import Any, List, Iterator


class MaskInput(object):
    def __init__(self, mask_size: int, batch_size: int = None):
        self.mask_size = mask_size
        self.batch_size = batch_size

    def __call__(
        self, samples: torch.Tensor, mask_value: torch.Tensor
    ) -> List[torch.Tensor]:
        masked_samples = []
        samples = torch.split(samples, 1, dim=0)
        for sample in samples:
            masked_samples.append(self.mask_sample(sample, mask_value))
        return masked_samples

    @abstractmethod
    def mask_sample(
        self, sample: torch.Tensor, mask_value: torch.Tensor
    ) -> Iterator[torch.Tensor]:
        """The mask value should have the same shape as the sample."""
        raise NotImplementedError

    @abstractmethod
    def assign_important_score(
        self,
        sample: torch.Tensor,
        importance: torch.Tensor,
        masked_indices: List[Any],
    ) -> torch.Tensor:
        """The mask value should have the same shape as the sample."""
        raise NotImplementedError


class RandomlyMaskInput(MaskInput):
    def __init__(self, seed, mask_size: int, batch_size: int = None):
        super().__init__(mask_size, batch_size)
        self.seed = seed
        self.random_generator = torch.Generator().manual_seed(self.seed)

    def mask_sample(
        self, sample: torch.Tensor, mask_value: torch.Tensor
    ) -> Iterator[torch.Tensor]:
        """The mask value should have the same shape as the sample."""
        # sample shape: 1, ...; mask_value shape: 1, ...

        masked_data = []
        masked_samples = []
        while True:
            binary_tensor = torch.randint(
                0,
                2,
                sample.size(),
                generator=self.random_generator,
            ).to(sample.device)
            masked_sample = mask_value + (sample - mask_value) * binary_tensor.float()

            masked_data.append(binary_tensor)
            masked_samples.append(masked_sample)

            if len(masked_samples) == self.batch_size:
                batch_masked_data = torch.cat(masked_data, dim=0)
                batch_masked_samples = torch.cat(masked_samples, dim=0)
                yield batch_masked_data, batch_masked_samples
                masked_data = []
                masked_samples = []


class MaskInputImage(MaskInput):
    def mask_sample(
        self, sample: torch.Tensor, mask_value: torch.Tensor
    ) -> Iterator[torch.Tensor]:
        *_, h, w = sample.shape  # 1,C,H,W

        masked_samples = []
        for i in range(ceil(h / self.mask_size)):
            for j in range(ceil(w / self.mask_size)):
                masked_sample = sample.detach().clone()
                masked_sample[
                    :,
                    :,
                    i * self.mask_size : (i + 1) * self.mask_size,
                    j * self.mask_size : (j + 1) * self.mask_size,
                ] = mask_value[
                    :,
                    :,
                    i * self.mask_size : (i + 1) * self.mask_size,
                    j * self.mask_size : (j + 1) * self.mask_size,
                ]

                masked_samples.append(masked_sample)

                if len(masked_samples) == self.batch_size:
                    batch_masked_samples = torch.cat(masked_samples, dim=0)
                    yield batch_masked_samples
                    masked_samples = []

        if len(masked_samples) > 0:
            batch_masked_samples = torch.cat(masked_samples, dim=0)
            yield batch_masked_samples

    def assign_important_score(
        self,
        sample: torch.Tensor,
        importance: torch.Tensor,
    ) -> torch.Tensor:
        *_, h, w = sample.shape  # 1,C,H,W
        sample_importance = importance.new_zeros((h, w), dtype=torch.float32)
        for i in range(ceil(h / self.mask_size)):
            for j in range(ceil(w / self.mask_size)):
                sample_importance[
                    i * self.mask_size : (i + 1) * self.mask_size,
                    j * self.mask_size : (j + 1) * self.mask_size,
                ] = importance[i * ceil(w / self.mask_size) + j]

        return sample_importance


class MaskInputSequence(MaskInput):
    def mask_sample(
        self, sample: torch.Tensor, mask_value: torch.Tensor
    ) -> Iterator[torch.Tensor]:
        _, seq_len = sample.shape  # 1,seq_len

        masked_samples = []
        for i in range(ceil(seq_len / self.mask_size)):
            masked_sample = sample.detach().clone()
            masked_sample[
                :,
                i * self.mask_size : (i + 1) * self.mask_size,
            ] = mask_value[
                :,
                i * self.mask_size : (i + 1) * self.mask_size,
            ]
            masked_samples.append(masked_sample)

            if len(masked_samples) == self.batch_size:
                batch_masked_samples = torch.cat(masked_samples, dim=0)
                yield batch_masked_samples
                masked_samples = []

        if len(masked_samples) > 0:
            batch_masked_samples = torch.cat(masked_samples, dim=0)
            yield batch_masked_samples

    def assign_important_score(
        self,
        sample: torch.Tensor,
        importance: torch.Tensor,
    ) -> torch.Tensor:
        _, seq_len = sample.shape  # 1,seq_len
        sample_importance = importance.new_zeros((seq_len,), dtype=torch.float32)
        for i in range(ceil(seq_len / self.mask_size)):
            sample_importance[i * self.mask_size : (i + 1) * self.mask_size,] = (
                importance[i]
            )

        return sample_importance
