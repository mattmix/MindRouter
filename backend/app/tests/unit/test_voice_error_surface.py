############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_voice_error_surface.py: Voice API error classification
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Whose fault is a failed voice request?

Lives in its own file rather than in test_voice_api.py because that file
contributes ZERO tests to the full-suite run — a pre-existing collection
failure — so guards written there protect nothing in CI. These are pure source
contracts with no heavy imports, so they collect everywhere.
"""

class TestEmptyAudioIsTheCallersError:
    """An unrecordable upload used to surface as `STT service error: 500`.

    That is upstream's status passed through verbatim, and it reads as a broken
    speech service — it sent an operator to check GPUs and backend health when
    the real cause was a recorder that stopped before flushing a chunk. The
    speech service is behaving correctly when it refuses audio it cannot
    decode; the caller is the one who needs to know.
    """

    def _src(self):
        from pathlib import Path
        return (Path(__file__).resolve().parents[2] / "api" / "voice_api.py").read_text()

    def test_empty_upload_is_rejected_before_dialing_out(self):
        """No point spending a backend round-trip to learn the body is empty."""
        src = self._src()
        i = src.index("audio_data = await file.read()")
        block = src[i:i + 1400]
        assert "if not audio_data:" in block
        assert "status_code=400" in block
        assert "No audio detected" in block
        # ...and before the upstream POST
        assert block.index("if not audio_data:") < src[i:].index("client.post")

    def test_only_a_truly_empty_body_is_size_rejected(self):
        """A short real clip can be tiny. Rejecting on a byte threshold would
        refuse legitimate recordings, so anything non-empty still goes upstream
        and is judged by the decoder."""
        src = self._src()
        i = src.index("audio_data = await file.read()")
        block = src[i:i + 1400]
        assert "len(audio_data) <" not in block, "no arbitrary size threshold"

    def test_upstream_decode_failure_becomes_a_400(self):
        """speaches reports undecodable audio as 415 on newer builds and a bare
        500 on the one deployed, so both must map to the same caller-facing
        400."""
        src = self._src()
        i = src.index("body_lc = (resp.text or \"\").lower()")
        block = src[i:i + 900]
        assert "resp.status_code == 415" in block
        assert '"decode audio" in body_lc' in block
        assert "status_code=400" in block

    def test_decode_failure_is_not_reported_as_a_service_error(self):
        """The recorded error must distinguish 'the audio was bad' from 'the
        service was bad', or the request log keeps blaming the backend."""
        src = self._src()
        assert "STT rejected the audio: could not decode" in src

    def test_a_real_service_failure_still_reports_502(self):
        """Only decode failures are reclassified — an actually-down backend must
        keep saying so."""
        src = self._src()
        assert 'raise HTTPException(status_code=502, detail="STT service error")' in src

    def test_failed_decode_still_costs_no_tokens(self):
        """The 2.9.10 billing rule: a failed voice call bills zero."""
        src = self._src()
        i = src.index("stt_api_proxy_error")
        block = src[i:i + 900]
        assert "token_cost=0" in block
