from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator, Sequence
from typing import TypeVar, overload

# Define a generic type variable to represent the type of elements in the sequences
T = TypeVar('T')


class ChainedSequence(Sequence[T]):
    """
    A memory-efficient, read-only sequence that concatenates any number of
    sequences.
    Inherits from collections.abc.Sequence to automatically provide standard
    methods like __iter__, __contains__, __reversed__, count(), and index().
    """

    __slots__ = ('_sequences', '_cumulative_lengths')

    def __init__(self, *sequences: Sequence[T]) -> None:
        self._sequences = sequences

        cumulative_lengths: list[int] = []
        total_length = 0
        for sequence in sequences:
            total_length += len(sequence)
            cumulative_lengths.append(total_length)

        self._cumulative_lengths = tuple(cumulative_lengths)

    # @overload decorators help static type checkers (like mypy) understand
    # the different return types based on the input type.
    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[T]: ...

    def __getitem__(self, index: int | slice) -> T | Sequence[T]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]

        total_len = len(self)

        if index < 0:
            index += total_len

        if index < 0 or index >= total_len:
            raise IndexError("ChainedSequence index out of range")

        sequence_index = bisect_right(self._cumulative_lengths, index)
        previous_length = 0 if sequence_index == 0 else self._cumulative_lengths[sequence_index - 1]
        return self._sequences[sequence_index][index - previous_length]

    def __len__(self) -> int:
        if not self._cumulative_lengths:
            return 0
        return self._cumulative_lengths[-1]

    def __iter__(self) -> Iterator[T]:
        for sequence in self._sequences:
            yield from sequence

    def __contains__(self, item: object) -> bool:
        return any(item in seq for seq in self._sequences)

    def __add__(self, other: Sequence[T]) -> ChainedSequence[T]:
        if not isinstance(other, Sequence):
            return NotImplemented
        return ChainedSequence(self, other)

    def __radd__(self, other: Sequence[T]) -> ChainedSequence[T]:
        if not isinstance(other, Sequence):
            return NotImplemented
        return ChainedSequence(other, self)


def chain_seqs(*sequences: Sequence[T]) -> ChainedSequence[T]:
    """Utility function to create a ChainedSequence from multiple sequences."""
    if not sequences:
        raise ValueError("At least one sequence must be provided")
    return ChainedSequence(*sequences)
