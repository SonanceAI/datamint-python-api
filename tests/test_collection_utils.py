import pytest

from datamint.utils.collection_utils import ChainedSequence, chain_seqs


def test_chained_sequence_supports_multiple_sequences() -> None:
    combined = ChainedSequence([1, 2], (3,), [4, 5])

    assert len(combined) == 5
    assert list(combined) == [1, 2, 3, 4, 5]
    assert combined[0] == 1
    assert combined[2] == 3
    assert combined[-1] == 5


def test_chained_sequence_slice_across_sequence_boundaries() -> None:
    combined = ChainedSequence([], [1, 2], (), [3, 4])

    assert combined[1:4] == [2, 3, 4]


def test_chain_seqs_returns_chained_sequence() -> None:
    combined = chain_seqs(["a"], ["b", "c"], ["d"])

    assert isinstance(combined, ChainedSequence)
    assert list(combined) == ["a", "b", "c", "d"]


def test_chain_seqs_requires_at_least_one_sequence() -> None:
    with pytest.raises(ValueError, match="At least one sequence must be provided"):
        chain_seqs()
