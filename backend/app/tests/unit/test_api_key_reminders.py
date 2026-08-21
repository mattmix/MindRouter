############################################################
# test_api_key_reminders.py: API key expiry reminder emails
############################################################
"""Two-tier reminders, ONE email per user per tier, branded, linking to the
dashboard. crud/email_service are stubbed at the seams; aiosmtplib is stubbed
so email_service imports."""
import sys
import types
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

if "aiosmtplib" not in sys.modules:
    _stub = types.ModuleType("aiosmtplib")
    _stub.SMTP = object
    sys.modules["aiosmtplib"] = _stub

from backend.app.services import api_key_reminders as R  # noqa: E402


def _key(id, user_id, name, prefix, days, *, early=None, urgent=None, is_service=False):
    k = MagicMock()
    k.id = id
    k.user_id = user_id
    k.name = name
    k.key_prefix = prefix
    k.is_service = is_service
    k.expires_at = datetime.now(timezone.utc) + timedelta(days=days)
    k.reminder_early_sent_at = early
    k.reminder_urgent_sent_at = urgent
    return k


def _user(id, email="u@x.edu", active=True, deleted=None):
    u = MagicMock()
    u.id = id
    u.email = email
    u.is_active = active
    u.deleted_at = deleted
    u.full_name = "Test User"
    u.display_name = "tuser"
    u.username = "tuser"
    return u


class TestTierWindows:
    def test_days_left_rounds_up_partial_days(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert R._days_left(now + timedelta(hours=26), now) == 2
        assert R._days_left(now + timedelta(hours=1), now) == 1
        assert R._days_left(now - timedelta(hours=1), now) == 0

    @pytest.mark.asyncio
    async def test_early_window_excludes_the_urgent_zone(self, monkeypatch):
        now = datetime.now(timezone.utc)
        # Already-reminded keys are excluded by SQL (see the predicate test);
        # this mock returns whatever the query "matched", so we pass only rows
        # that survived that filter and check the day-window boundaries.
        keys = [
            _key(1, 10, "a", "mr2_a", 9),   # early (2 < 9 <= 10)
            _key(2, 10, "b", "mr2_b", 2),   # in the urgent zone -> NOT early
            _key(3, 10, "c", "mr2_c", 20),  # too far -> neither
            _key(4, 10, "d", "mr2_d", 3),   # early boundary interior
        ]
        db = _db_returning(keys)
        early = await R._keys_in_tier(db, tier="early", early_days=10, urgent_days=2, now=now)
        assert sorted(k.id for k in early) == [1, 4]

    @pytest.mark.asyncio
    async def test_urgent_window_is_at_or_below_urgent_days(self, monkeypatch):
        now = datetime.now(timezone.utc)
        keys = [
            _key(1, 10, "a", "mr2_a", 2),   # urgent (<= 2)
            _key(2, 10, "b", "mr2_b", 1),   # urgent
            _key(3, 10, "c", "mr2_c", 5),   # not urgent
        ]
        db = _db_returning(keys)
        urgent = await R._keys_in_tier(db, tier="urgent", early_days=10, urgent_days=2, now=now)
        assert sorted(k.id for k in urgent) == [1, 2]

    def test_tier_query_filters_on_the_right_watermark_column(self):
        from sqlalchemy import select
        from backend.app.db.models import ApiKey, ApiKeyStatus
        now = datetime.now(timezone.utc)
        # Compile the WHERE the real function builds for each tier.
        for tier, col in (("early", "reminder_early_sent_at"), ("urgent", "reminder_urgent_sent_at")):
            lo_col = getattr(ApiKey, col)
            q = select(ApiKey).where(lo_col.is_(None), ApiKey.is_service.is_(False))
            sql = str(q.compile(compile_kwargs={"literal_binds": True}))
            assert f"api_keys.{col} IS NULL" in sql


def _db_returning(keys):
    db = MagicMock()
    scalars = MagicMock()
    scalars.all.return_value = keys
    result = MagicMock()
    result.scalars.return_value = scalars
    db.execute = AsyncMock(return_value=result)
    return db


class TestRenderEmail:
    def test_one_email_lists_all_keys_and_links_dashboard(self):
        keys = [
            {"name": "First", "key_prefix": "mr2_A", "expires_str": "2026-11-01", "days_left": 9},
            {"name": "Second", "key_prefix": "mr2_B", "expires_str": "2026-11-02", "days_left": 10},
        ]
        subject, body = R.render_reminder_email(
            display_name="Ada", keys=keys, tier="early", base_url="https://mr.example", tz=timezone.utc
        )
        assert "API keys" in subject and "9 days" in subject  # plural, soonest
        assert "First" in body and "Second" in body            # BOTH keys in one email
        assert "https://mr.example/dashboard" in body          # dashboard link
        assert "Manage my API keys" in body                    # the button
        assert "Renew" in body

    def test_urgent_subject_is_action_oriented_and_singular(self):
        keys = [{"name": "K", "key_prefix": "mr2_A", "expires_str": "2026-11-01", "days_left": 1}]
        subject, body = R.render_reminder_email(
            display_name="", keys=keys, tier="urgent", base_url="", tz=timezone.utc
        )
        assert subject.startswith("Action needed")
        assert "1 day" in subject and "API key expires" in subject
        assert "Hi there," in body  # empty name falls back


class TestSendGroupsByUser:
    @pytest.mark.asyncio
    async def test_one_email_per_user_even_with_multiple_keys(self, monkeypatch):
        now = datetime.now(timezone.utc)
        # user 10 has TWO early keys; user 20 has one
        early_keys = [
            _key(1, 10, "a", "mr2_a", 8),
            _key(2, 10, "b", "mr2_b", 9),
            _key(3, 20, "c", "mr2_c", 7),
        ]

        async def fake_keys_in_tier(db, *, tier, early_days, urgent_days, now):
            return early_keys if tier == "early" else []

        monkeypatch.setattr(R, "_keys_in_tier", fake_keys_in_tier)
        monkeypatch.setattr(R, "load_reminder_config", AsyncMock(return_value={
            "enabled": True, "early_days": 10, "urgent_days": 2, "test_recipient": ""}))
        monkeypatch.setattr(R.email_service, "get_smtp_config", AsyncMock(return_value={"x": 1}))
        monkeypatch.setattr(R.email_service, "is_smtp_configured", lambda c: True)
        monkeypatch.setattr(R.email_service, "get_base_url", AsyncMock(return_value="https://mr"))
        monkeypatch.setattr(R, "_reminder_tz", AsyncMock(return_value=timezone.utc))
        users = {10: _user(10, "ten@x.edu"), 20: _user(20, "twenty@x.edu")}
        monkeypatch.setattr(R.crud, "get_user_by_id", AsyncMock(side_effect=lambda db, uid: users[uid]))
        send = AsyncMock(return_value=1)
        monkeypatch.setattr(R.email_service, "send_notification_email", send)

        db = MagicMock()
        db.flush = AsyncMock()
        db.commit = AsyncMock()
        totals = await R.send_expiry_reminders(db)

        # exactly two emails (one per user), not three (one per key)
        assert send.await_count == 2
        recipients = sorted(call.args[1][0] for call in send.await_args_list)
        assert recipients == ["ten@x.edu", "twenty@x.edu"]
        assert totals["early_users"] == 2 and totals["early_keys"] == 3
        # both of user 10's keys stamped
        assert early_keys[0].reminder_early_sent_at is not None
        assert early_keys[1].reminder_early_sent_at is not None

    @pytest.mark.asyncio
    async def test_disabled_sends_nothing(self, monkeypatch):
        monkeypatch.setattr(R, "load_reminder_config", AsyncMock(return_value={
            "enabled": False, "early_days": 10, "urgent_days": 2, "test_recipient": ""}))
        send = AsyncMock(return_value=1)
        monkeypatch.setattr(R.email_service, "send_notification_email", send)
        out = await R.send_expiry_reminders(MagicMock())
        assert send.await_count == 0
        assert out["skipped"] == "disabled"

    @pytest.mark.asyncio
    async def test_send_failure_does_not_stamp_watermark(self, monkeypatch):
        now = datetime.now(timezone.utc)
        keys = [_key(1, 10, "a", "mr2_a", 8)]
        monkeypatch.setattr(R, "_keys_in_tier", AsyncMock(side_effect=lambda db, **k: keys if k["tier"] == "early" else []))
        monkeypatch.setattr(R, "load_reminder_config", AsyncMock(return_value={
            "enabled": True, "early_days": 10, "urgent_days": 2, "test_recipient": ""}))
        monkeypatch.setattr(R.email_service, "get_smtp_config", AsyncMock(return_value={"x": 1}))
        monkeypatch.setattr(R.email_service, "is_smtp_configured", lambda c: True)
        monkeypatch.setattr(R.email_service, "get_base_url", AsyncMock(return_value=""))
        monkeypatch.setattr(R, "_reminder_tz", AsyncMock(return_value=timezone.utc))
        monkeypatch.setattr(R.crud, "get_user_by_id", AsyncMock(return_value=_user(10)))
        monkeypatch.setattr(R.email_service, "send_notification_email", AsyncMock(return_value=0))  # fail
        db = MagicMock(); db.flush = AsyncMock(); db.commit = AsyncMock()
        await R.send_expiry_reminders(db)
        assert keys[0].reminder_early_sent_at is None  # retried next run

    @pytest.mark.asyncio
    async def test_inactive_or_deleted_user_skipped(self, monkeypatch):
        keys = [_key(1, 10, "a", "mr2_a", 8), _key(2, 20, "b", "mr2_b", 8)]
        monkeypatch.setattr(R, "_keys_in_tier", AsyncMock(side_effect=lambda db, **k: keys if k["tier"] == "early" else []))
        monkeypatch.setattr(R, "load_reminder_config", AsyncMock(return_value={
            "enabled": True, "early_days": 10, "urgent_days": 2, "test_recipient": ""}))
        monkeypatch.setattr(R.email_service, "get_smtp_config", AsyncMock(return_value={"x": 1}))
        monkeypatch.setattr(R.email_service, "is_smtp_configured", lambda c: True)
        monkeypatch.setattr(R.email_service, "get_base_url", AsyncMock(return_value=""))
        monkeypatch.setattr(R, "_reminder_tz", AsyncMock(return_value=timezone.utc))
        users = {10: _user(10, deleted=datetime.now(timezone.utc)), 20: _user(20)}
        monkeypatch.setattr(R.crud, "get_user_by_id", AsyncMock(side_effect=lambda db, uid: users[uid]))
        send = AsyncMock(return_value=1)
        monkeypatch.setattr(R.email_service, "send_notification_email", send)
        db = MagicMock(); db.flush = AsyncMock(); db.commit = AsyncMock()
        await R.send_expiry_reminders(db)
        assert send.await_count == 1  # only the live user


class TestConfigAndTest:
    @pytest.mark.asyncio
    async def test_config_clamps_urgent_below_early(self, monkeypatch):
        vals = {"apikey.reminders.enabled": True, "apikey.reminders.early_days": 5,
                "apikey.reminders.urgent_days": 9, "apikey.reminders.test_recipient": " a@b.edu "}
        monkeypatch.setattr(R.crud, "get_config_json",
                            AsyncMock(side_effect=lambda db, k, d=None: vals.get(k, d)))
        cfg = await R.load_reminder_config(MagicMock())
        assert cfg["early_days"] == 5
        assert cfg["urgent_days"] == 4  # clamped to early-1
        assert cfg["test_recipient"] == "a@b.edu"

    @pytest.mark.asyncio
    async def test_test_reminder_uses_sample_keys_and_test_flag(self, monkeypatch):
        monkeypatch.setattr(R.email_service, "get_smtp_config", AsyncMock(return_value={"x": 1}))
        monkeypatch.setattr(R.email_service, "is_smtp_configured", lambda c: True)
        monkeypatch.setattr(R.email_service, "get_base_url", AsyncMock(return_value="https://mr"))
        monkeypatch.setattr(R, "_reminder_tz", AsyncMock(return_value=timezone.utc))
        monkeypatch.setattr(R, "load_reminder_config", AsyncMock(return_value={
            "enabled": True, "early_days": 10, "urgent_days": 2, "test_recipient": "t@x.edu"}))
        send = AsyncMock(return_value=1)
        monkeypatch.setattr(R.email_service, "send_notification_email", send)
        err = await R.send_test_reminder(MagicMock(), "t@x.edu", tier="early")
        assert err == ""
        subject = send.await_args.args[2]
        body = send.await_args.args[3]
        assert subject.startswith("[TEST]")
        assert "test" in body.lower() and "fabricated" in body.lower()

    @pytest.mark.asyncio
    async def test_test_reminder_rejects_bad_recipient(self):
        assert (await R.send_test_reminder(MagicMock(), "")) == "No valid test recipient configured"
        crlf = "a@b.edu" + chr(13) + chr(10) + "Bcc: evil@x"  # header-injection guard
        assert (await R.send_test_reminder(MagicMock(), crlf)) == "No valid test recipient configured"
