import h5py
import numpy as np
import pytest

from scida import discovertypes
from scida.customs.arepo.series import ArepoSimulation
from scida.customs.swift.dataset import SwiftSnapshot
from scida.discovertypes import (
    CandidateStatus,
    Confidence,
    DetectionResult,
    Specificity,
    _determine_type,
    is_valid_candidate,
)


@pytest.mark.unit
class TestCandidateStatus:
    def test_enum_values(self):
        assert CandidateStatus.NO.value == 0
        assert CandidateStatus.MAYBE.value == 1
        assert CandidateStatus.YES.value == 2

    def test_ordering(self):
        assert CandidateStatus.NO.value < CandidateStatus.MAYBE.value
        assert CandidateStatus.MAYBE.value < CandidateStatus.YES.value


@pytest.mark.unit
class TestIsValidCandidate:
    def test_bool_true(self):
        result = is_valid_candidate(True)
        assert result == CandidateStatus.MAYBE

    def test_bool_false(self):
        result = is_valid_candidate(False)
        assert result == CandidateStatus.NO

    def test_candidate_status_passthrough(self):
        assert is_valid_candidate(CandidateStatus.YES) == CandidateStatus.YES
        assert is_valid_candidate(CandidateStatus.NO) == CandidateStatus.NO
        assert is_valid_candidate(CandidateStatus.MAYBE) == CandidateStatus.MAYBE


@pytest.mark.unit
class TestDetectionResolution:
    def test_confidence_beats_specificity(self, monkeypatch):
        class StructurallyCompatibleSpecificDataset:
            @classmethod
            def validate_path(cls, path, **kwargs):
                return DetectionResult.match(
                    Confidence.GENERIC_HEADER,
                    specificity=Specificity.SUBFAMILY,
                    evidence=("generic header",),
                )

        class FormatIdentifiedFamilyDataset:
            @classmethod
            def validate_path(cls, path, **kwargs):
                return DetectionResult.match(
                    Confidence.FORMAT_MARKER,
                    specificity=Specificity.FAMILY,
                    evidence=("format marker",),
                )

        monkeypatch.setattr(
            discovertypes,
            "dataset_type_registry",
            {
                "StructurallyCompatibleSpecificDataset": (
                    StructurallyCompatibleSpecificDataset
                ),
                "FormatIdentifiedFamilyDataset": FormatIdentifiedFamilyDataset,
            },
        )
        monkeypatch.setattr(discovertypes, "dataseries_type_registry", {})

        names, classes = _determine_type("dummy")

        assert names == ["FormatIdentifiedFamilyDataset"]
        assert classes == [FormatIdentifiedFamilyDataset]

    def test_specificity_breaks_equal_confidence_tie(self, monkeypatch):
        class FamilyDataset:
            @classmethod
            def validate_path(cls, path, **kwargs):
                return DetectionResult.match(
                    Confidence.FORMAT_MARKER,
                    specificity=Specificity.FAMILY,
                    evidence=("format marker",),
                )

        class SubfamilyDataset:
            @classmethod
            def validate_path(cls, path, **kwargs):
                return DetectionResult.match(
                    Confidence.FORMAT_MARKER,
                    specificity=Specificity.SUBFAMILY,
                    evidence=("format marker",),
                )

        monkeypatch.setattr(
            discovertypes,
            "dataset_type_registry",
            {
                "FamilyDataset": FamilyDataset,
                "SubfamilyDataset": SubfamilyDataset,
            },
        )
        monkeypatch.setattr(discovertypes, "dataseries_type_registry", {})

        names, classes = _determine_type("dummy")

        assert names == ["SubfamilyDataset"]
        assert classes == [SubfamilyDataset]

    def test_swift_marker_beats_generic_arepo_series_shape(self, monkeypatch, tmp_path):
        for i in range(2):
            with h5py.File(tmp_path / f"flamingo_0042.{i}.hdf5", "w") as f:
                header = f.create_group("Header")
                header.attrs["NumFilesPerSnapshot"] = 1
                header.attrs["NumPart_ThisFile"] = np.zeros(6, dtype="u8")
                header.attrs["NumPart_Total"] = np.zeros(6, dtype="u8")
                code = f.create_group("Code")
                code.attrs["Code"] = np.bytes_(b"SWIFT")

        monkeypatch.setattr(
            discovertypes,
            "dataset_type_registry",
            {"SwiftSnapshot": SwiftSnapshot},
        )
        monkeypatch.setattr(
            discovertypes,
            "dataseries_type_registry",
            {"ArepoSimulation": ArepoSimulation},
        )

        names, classes = _determine_type(tmp_path)

        assert names == ["SwiftSnapshot"]
        assert classes == [SwiftSnapshot]
