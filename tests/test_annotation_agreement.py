import numpy as np
import pytest

from datamint.entities.annotations import BoxAnnotation, ImageClassification, ImageSegmentation
from datamint.utils.annotation_agreement import (
    cohen_kappa,
    compute_agreement,
    dice_coefficient,
    fleiss_kappa,
    iou_boxes,
)


def _segmentation(mask: np.ndarray, resource_id: str, created_by: str, identifier: str = "lesion"):
    return ImageSegmentation(mask=mask, name=identifier, resource_id=resource_id, created_by=created_by)


def _box(point1, point2, resource_id: str, created_by: str, identifier: str = "tumor", coords_system="pixel"):
    return BoxAnnotation.from_points(
        point1, point2, identifier=identifier, resource_id=resource_id,
        created_by=created_by, coords_system=coords_system,
    )


def _classification(value: str, resource_id: str, created_by: str, identifier: str = "finding"):
    return ImageClassification(name=identifier, value=value, resource_id=resource_id, created_by=created_by)


class TestDiceCoefficient:
    def test_identical_masks(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:6, 2:6] = True
        assert dice_coefficient(mask, mask.copy()) == pytest.approx(1.0)

    def test_partial_overlap(self):
        mask_a = np.zeros((10, 10), dtype=np.uint8)
        mask_a[2:6, 2:6] = 1  # 16 px
        mask_b = np.zeros((10, 10), dtype=np.uint8)
        mask_b[2:6, 2:8] = 1  # 24 px, overlap 16 px
        # dice = 2*16 / (16+24) = 0.8
        assert dice_coefficient(mask_a, mask_b) == pytest.approx(0.8)

    def test_both_empty_is_perfect_agreement(self):
        mask = np.zeros((5, 5), dtype=bool)
        assert dice_coefficient(mask, mask.copy()) == pytest.approx(1.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shapes differ"):
            dice_coefficient(np.zeros((5, 5)), np.zeros((6, 6)))


class TestIouBoxes:
    def test_partial_overlap(self):
        box_a = _box((0, 0), (10, 10), "r1", "a@x.com").geometry
        box_b = _box((5, 5), (15, 15), "r1", "b@x.com").geometry
        # intersection 5x5=25, union 100+100-25=175
        assert iou_boxes(box_a, box_b) == pytest.approx(25 / 175)

    def test_mismatched_coordinate_system_raises(self):
        box_a = _box((0, 0), (10, 10), "r1", "a@x.com", coords_system="pixel").geometry
        box_b = BoxAnnotation(
            geometry={"points": [(0, 0, 0), (0, 10, 0), (10, 0, 0), (10, 10, 0)],
                      "coordinate_system": "patient"},
            identifier="tumor", resource_id="r1", created_by="b@x.com",
        ).geometry
        with pytest.raises(ValueError, match="coordinate system"):
            iou_boxes(box_a, box_b)


class TestCohenKappa:
    def test_known_value(self):
        labels_a = ["yes", "yes", "no", "no"]
        labels_b = ["yes", "no", "no", "no"]
        assert cohen_kappa(labels_a, labels_b) == pytest.approx(0.5)

    def test_perfect_agreement(self):
        labels = ["a", "b", "a", "c"]
        assert cohen_kappa(labels, labels) == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            cohen_kappa(["a"], ["a", "b"])


class TestFleissKappa:
    def test_perfect_agreement(self):
        ratings = [["A", "A", "A"], ["B", "B", "B"], ["A", "A", "A"]]
        assert fleiss_kappa(ratings) == pytest.approx(1.0)

    def test_ragged_raters_raises(self):
        with pytest.raises(ValueError, match="same number"):
            fleiss_kappa([["A", "B", "A"], ["A", "B"]])


class TestComputeAgreementSegmentation:
    def test_dice_end_to_end(self):
        mask_a = np.zeros((10, 10), dtype=np.uint8)
        mask_a[2:6, 2:6] = 1
        mask_b = np.zeros((10, 10), dtype=np.uint8)
        mask_b[2:6, 2:8] = 1

        result = compute_agreement([
            _segmentation(mask_a, "r1", "alice@x.com"),
            _segmentation(mask_b, "r1", "bob@x.com"),
        ])

        assert result.overall == pytest.approx(0.8)
        assert len(result.per_pair) == 1
        assert result.per_pair.iloc[0]["metric"] == "dice"

    def test_threshold_flags_low_agreement_resource(self):
        mask_a = np.zeros((10, 10), dtype=np.uint8)
        mask_a[2:6, 2:6] = 1
        mask_b = np.zeros((10, 10), dtype=np.uint8)
        mask_b[2:6, 2:8] = 1

        result = compute_agreement([
            _segmentation(mask_a, "r1", "alice@x.com"),
            _segmentation(mask_b, "r1", "bob@x.com"),
        ], threshold=0.99)

        assert len(result.flagged) == 1

    def test_no_threshold_returns_empty_flagged(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0:2, 0:2] = 1
        result = compute_agreement([
            _segmentation(mask, "r1", "alice@x.com"),
            _segmentation(mask, "r1", "bob@x.com"),
        ])
        assert result.flagged.empty

    def test_single_annotator_raises(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="2\\+ distinct annotators"):
            compute_agreement([_segmentation(mask, "r1", "alice@x.com")])


class TestComputeAgreementBox:
    def test_iou_end_to_end(self):
        result = compute_agreement([
            _box((0, 0), (10, 10), "r2", "alice@x.com"),
            _box((5, 5), (15, 15), "r2", "bob@x.com"),
        ])
        assert result.overall == pytest.approx(25 / 175)


class TestComputeAgreementCategorical:
    def test_cohen_kappa_two_annotators(self):
        result = compute_agreement([
            _classification("benign", "r3", "alice@x.com"),
            _classification("malignant", "r3", "bob@x.com"),
            _classification("benign", "r4", "alice@x.com"),
            _classification("benign", "r4", "bob@x.com"),
        ])
        assert result.overall == pytest.approx(0.0)
        assert (result.per_pair["metric"] == "cohen_kappa").all()

    def test_fleiss_kappa_three_annotators(self):
        annotations = []
        for resource_id, value in [("r1", "A"), ("r2", "B"), ("r3", "A")]:
            for who in ["a", "b", "c"]:
                annotations.append(_classification(value, resource_id, f"{who}@x.com"))
        result = compute_agreement(annotations)
        assert result.overall == pytest.approx(1.0)

    def test_explicit_cohen_kappa_with_three_annotators_raises(self):
        annotations = []
        for resource_id, value in [("r1", "A"), ("r2", "B")]:
            for who in ["a", "b", "c"]:
                annotations.append(_classification(value, resource_id, f"{who}@x.com"))
        with pytest.raises(ValueError, match="requires exactly 2 annotators"):
            compute_agreement(annotations, metric="cohen_kappa")

    def test_fleiss_marks_partial_coverage_items_and_excludes_them_from_overall(self):
        annotations = []
        # r1: all 3 annotators rate it, all agree -> counts toward Fleiss' kappa
        for who in ["a", "b", "c"]:
            annotations.append(_classification("A", "r1", f"{who}@x.com"))
        # r2: only 2 of 3 annotators rated it, and they disagree
        annotations.append(_classification("A", "r2", "a@x.com"))
        annotations.append(_classification("B", "r2", "b@x.com"))

        result = compute_agreement(annotations)

        # overall only reflects r1, the fully-rated item
        assert result.overall == pytest.approx(1.0)

        by_resource = result.per_resource_mean.set_index("resource_id")
        assert bool(by_resource.loc["r1", "used_in_overall"]) is True
        assert bool(by_resource.loc["r2", "used_in_overall"]) is False
        assert by_resource.loc["r2", "score"] == pytest.approx(0.0)

    def test_fleiss_kappa_with_rotating_annotator_pool(self):
        # 4 annotators total, but each resource is only rated by 3 of them on
        # rotation. Fleiss' kappa only needs a consistent *count* of raters
        # per item, not the same identities, so this should not raise.
        annotations = []
        trios = [
            ("r1", {"a": "A", "b": "A", "c": "A"}),
            ("r2", {"a": "A", "b": "B", "d": "A"}),
            ("r3", {"b": "A", "c": "A", "d": "A"}),
        ]
        for resource_id, ratings in trios:
            for who, value in ratings.items():
                annotations.append(_classification(value, resource_id, f"{who}@x.com"))

        result = compute_agreement(annotations)

        assert (result.per_pair["used_in_overall"]).all()
        by_resource = result.per_resource_mean.set_index("resource_id")
        assert bool(by_resource.loc["r1", "used_in_overall"]) is True
        assert bool(by_resource.loc["r2", "used_in_overall"]) is True
        assert bool(by_resource.loc["r3", "used_in_overall"]) is True

    def test_fleiss_kappa_excludes_minority_rater_count(self):
        # 3 items rated by 3 annotators (the common case), 1 item rated by
        # only 2 -> the 2-rater item should be excluded from overall.
        annotations = []
        for resource_id in ["r1", "r2", "r3"]:
            for who in ["a", "b", "c"]:
                annotations.append(_classification("A", resource_id, f"{who}@x.com"))
        annotations.append(_classification("A", "r4", "a@x.com"))
        annotations.append(_classification("B", "r4", "b@x.com"))

        result = compute_agreement(annotations)

        by_resource = result.per_resource_mean.set_index("resource_id")
        assert bool(by_resource.loc["r4", "used_in_overall"]) is False
        assert result.overall == pytest.approx(1.0)


class TestComputeAgreementValidation:
    def test_mixed_kinds_without_explicit_metric_raises(self):
        box = _box((0, 0), (10, 10), "r5", "alice@x.com")
        classification = _classification("benign", "r5", "alice@x.com")
        with pytest.raises(ValueError, match="mixed/unsupported annotation kinds"):
            compute_agreement([box, classification])

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="No annotations provided"):
            compute_agreement([])
