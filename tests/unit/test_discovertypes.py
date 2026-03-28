import pytest

from scida.discovertypes import CandidateStatus, is_valid_candidate


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
