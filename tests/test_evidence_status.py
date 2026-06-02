import api

class DummyEvidence:
    def __init__(self, decision):
        self.decision = decision


def test_format_evidence_status_mappings():
    assert api._format_evidence_status(DummyEvidence('ok')) == 'Đủ bằng chứng'
    assert api._format_evidence_status(DummyEvidence('CONFLICT')) == 'Có mâu thuẫn'
    assert api._format_evidence_status(DummyEvidence('abstain')) == 'Thiếu bằng chứng'
    # Handle empty and None decisions
    assert api._format_evidence_status(DummyEvidence('')) == 'Không rõ'
    assert api._format_evidence_status(DummyEvidence(None)) == 'Không rõ'
    # Handle None evidence object
    assert api._format_evidence_status(None) == 'Không rõ'
