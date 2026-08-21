############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# api_key_reminders.py: expiry reminder emails for personal
#     API keys. Two tiers (early / urgent), each sent at most
#     once per key per period; re-armed on renewal. Branded
#     via email_service, linking to the user's dashboard.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################
"""API key expiry reminder emails.

A tier is a (window, watermark) pair:
  - early : key expires within ``early_days`` and > ``urgent_days`` away,
            ``reminder_early_sent_at`` still NULL
  - urgent: key expires within ``urgent_days`` (and not yet expired),
            ``reminder_urgent_sent_at`` still NULL

One email per (user, tier) run lists every one of that user's keys in the
tier and links to the dashboard, where they Renew/Delete. The watermark is
stamped only on keys whose email actually went out, so a send failure is
retried next cycle rather than silently dropped. Service keys never expire
and are excluded; expired keys are past reminding.
"""

import html as _html
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from backend.app.db import crud
from backend.app.db.models import ApiKey, ApiKeyStatus, User
from backend.app.logging_config import get_logger
from backend.app.services import branding as _branding
from backend.app.services import email_service

logger = get_logger(__name__)

DEFAULT_ENABLED = False
DEFAULT_EARLY_DAYS = 10
DEFAULT_URGENT_DAYS = 2


async def load_reminder_config(db) -> Dict[str, Any]:
    """Read the admin reminder settings (app_config), with safe fallbacks."""
    def _int(v, default):
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    enabled = bool(await crud.get_config_json(db, "apikey.reminders.enabled", DEFAULT_ENABLED))
    early = _int(await crud.get_config_json(db, "apikey.reminders.early_days", DEFAULT_EARLY_DAYS), DEFAULT_EARLY_DAYS)
    urgent = _int(await crud.get_config_json(db, "apikey.reminders.urgent_days", DEFAULT_URGENT_DAYS), DEFAULT_URGENT_DAYS)
    test_to = await crud.get_config_json(db, "apikey.reminders.test_recipient", "") or ""
    # urgent must be strictly below early so the two windows don't overlap.
    early = max(1, early)
    urgent = max(0, min(urgent, early - 1))
    return {
        "enabled": enabled,
        "early_days": early,
        "urgent_days": urgent,
        "test_recipient": str(test_to).strip(),
    }


def _days_left(expires_at: datetime, now: datetime) -> int:
    """Whole days until expiry (ceil-ish: same-day but future counts as today)."""
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    delta = expires_at - now
    # round up partial days so "in 26 hours" reads as "1 day", not "0".
    return max(0, -(-int(delta.total_seconds()) // 86400))


async def _keys_in_tier(db, *, tier: str, early_days: int, urgent_days: int, now: datetime):
    """Active, non-service, unexpired keys due for the given tier's email."""
    lo_col = ApiKey.reminder_early_sent_at if tier == "early" else ApiKey.reminder_urgent_sent_at
    q = (
        select(ApiKey)
        .where(
            ApiKey.is_service.is_(False),
            ApiKey.status == ApiKeyStatus.ACTIVE,
            ApiKey.expires_at.isnot(None),
            ApiKey.expires_at > now,
            lo_col.is_(None),
        )
        .order_by(ApiKey.user_id, ApiKey.expires_at.asc())
    )
    result = await db.execute(q)
    keys = list(result.scalars().all())
    out = []
    for k in keys:
        d = _days_left(k.expires_at, now)
        if tier == "urgent":
            if d <= urgent_days:
                out.append(k)
        else:  # early
            if urgent_days < d <= early_days:
                out.append(k)
    return out


def _fmt_date(dt: datetime, tz) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz).strftime("%Y-%m-%d")


def render_reminder_email(
    *, display_name: str, keys: List[Dict[str, Any]], tier: str, base_url: str, tz
) -> tuple:
    """Return (subject, body_html) for one user's reminder. ``keys`` is a list
    of dicts: {name, key_prefix, expires_at, days_left}."""
    app_name = _branding.get_branding()["app_name"]
    n = len(keys)
    noun = "API key" if n == 1 else "API keys"
    soonest = min(k["days_left"] for k in keys)

    if tier == "urgent":
        subject = f"Action needed: your {app_name} {noun} expire{'s' if n == 1 else ''} in {soonest} day{'s' if soonest != 1 else ''}"
        lead = (
            f"One or more of your {app_name} {noun} will expire very soon. "
            "When a key expires it stops working for API calls and for the web chat, "
            "image, and video tools. You can keep using the same key by renewing it — "
            "the key value does not change, so nothing in your applications needs updating."
        )
    else:
        subject = f"Reminder: your {app_name} {noun} expire{'s' if n == 1 else ''} in {soonest} days"
        lead = (
            f"This is a friendly heads-up that one or more of your {app_name} {noun} "
            "will expire soon. You can renew a key at any time — the key value stays the "
            "same, so renewing never requires changing anything in your applications."
        )

    rows = []
    for k in keys:
        dl = k["days_left"]
        dl_txt = f"{dl} day{'s' if dl != 1 else ''}"
        rows.append(
            f"<tr><td>{_html.escape(k['name'])}</td>"
            f"<td><code>{_html.escape(k['key_prefix'])}…</code></td>"
            f"<td>{_html.escape(k['expires_str'])}</td>"
            f"<td>{dl_txt}</td></tr>"
        )
    accent_ink = _branding.get_branding()["primary_light_ink"]
    dash = f"{base_url}/dashboard" if base_url else "/dashboard"
    button = (
        f'<p style="margin:24px 0;"><a href="{dash}" '
        f'style="background:{accent_ink};color:#ffffff;text-decoration:none;'
        f'padding:12px 22px;border-radius:6px;font-weight:600;display:inline-block;">'
        f'Manage my API keys</a></p>'
    )
    body = (
        f"<p>Hi {_html.escape(display_name or 'there')},</p>"
        f"<p>{lead}</p>"
        f'<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">'
        f"<tr><th>Key</th><th>Prefix</th><th>Expires</th><th>Time left</th></tr>"
        f"{''.join(rows)}</table>"
        f"{button}"
        f'<p>Open your <a href="{dash}" style="color:{accent_ink};">dashboard</a> to '
        f"<strong>Renew</strong> a key for another period, or <strong>Revoke</strong> one "
        f"you no longer need. Renewing keeps the same key value.</p>"
        f"<p style=\"color:#888;font-size:13px;\">Service keys are managed by administrators "
        f"and never expire — this reminder only covers your personal keys.</p>"
    )
    return subject, body


async def send_expiry_reminders(db, *, dry_run: bool = False) -> Dict[str, int]:
    """Send due reminder emails (both tiers). Returns per-tier counts.

    dry_run computes and returns the counts without sending or stamping.
    """
    cfg = await load_reminder_config(db)
    if not cfg["enabled"]:
        return {"early_users": 0, "early_keys": 0, "urgent_users": 0, "urgent_keys": 0, "skipped": "disabled"}

    smtp = await email_service.get_smtp_config(db)
    if not email_service.is_smtp_configured(smtp):
        logger.warning("apikey_reminder_smtp_not_configured")
        return {"early_users": 0, "early_keys": 0, "urgent_users": 0, "urgent_keys": 0, "skipped": "no_smtp"}

    base_url = await email_service.get_base_url(db)
    tz = await _reminder_tz(db)
    now = datetime.now(timezone.utc)
    totals = {"early_users": 0, "early_keys": 0, "urgent_users": 0, "urgent_keys": 0}

    for tier in ("urgent", "early"):  # urgent first: the closer deadline wins the run
        keys = await _keys_in_tier(
            db, tier=tier, early_days=cfg["early_days"], urgent_days=cfg["urgent_days"], now=now
        )
        by_user: Dict[int, List[ApiKey]] = {}
        for k in keys:
            by_user.setdefault(k.user_id, []).append(k)

        for user_id, user_keys in by_user.items():
            user = await crud.get_user_by_id(db, user_id)
            if not user or not user.is_active or user.deleted_at or not user.email:
                continue
            rendered = [
                {
                    "name": k.name,
                    "key_prefix": k.key_prefix,
                    "expires_str": _fmt_date(k.expires_at, tz),
                    "days_left": _days_left(k.expires_at, now),
                }
                for k in user_keys
            ]
            subject, body = render_reminder_email(
                display_name=user.full_name or user.display_name or user.username,
                keys=rendered, tier=tier, base_url=base_url, tz=tz,
            )
            totals[f"{tier}_users"] += 1
            totals[f"{tier}_keys"] += len(user_keys)
            if dry_run:
                continue
            sent = await email_service.send_notification_email(
                smtp, [user.email], subject, body, base_url=base_url
            )
            if sent:
                stamp = now
                for k in user_keys:
                    if tier == "urgent":
                        k.reminder_urgent_sent_at = stamp
                    else:
                        k.reminder_early_sent_at = stamp
                await db.flush()
                logger.info(
                    "apikey_reminder_sent", tier=tier, user_id=user_id, keys=len(user_keys)
                )
            else:
                logger.warning("apikey_reminder_send_failed", tier=tier, user_id=user_id)
    if not dry_run:
        await db.commit()
    return totals


async def _reminder_tz(db):
    import zoneinfo
    name = await crud.get_config_json(db, "app.timezone", "America/Los_Angeles") or "America/Los_Angeles"
    try:
        return zoneinfo.ZoneInfo(str(name))
    except Exception:
        return zoneinfo.ZoneInfo("America/Los_Angeles")


async def send_test_reminder(db, recipient: str, tier: str = "early") -> str:
    """Send a sample reminder to ``recipient`` (the admin's test address).
    Returns "" on success or an error string. Uses fabricated sample keys so
    no real key state is touched."""
    recipient = (recipient or "").strip()
    if not recipient or "\n" in recipient or "\r" in recipient:
        return "No valid test recipient configured"
    smtp = await email_service.get_smtp_config(db)
    if not email_service.is_smtp_configured(smtp):
        return "SMTP is not configured"
    base_url = await email_service.get_base_url(db)
    tz = await _reminder_tz(db)
    cfg = await load_reminder_config(db)
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    days = cfg["urgent_days"] if tier == "urgent" else cfg["early_days"]
    sample = [
        {
            "name": "Example Key",
            "key_prefix": "mr2_XXXXXXXX",
            "expires_str": _fmt_date(now + timedelta(days=days), tz),
            "days_left": max(1, days),
        },
        {
            "name": "power automate",
            "key_prefix": "mr2_YYYYYYYY",
            "expires_str": _fmt_date(now + timedelta(days=max(1, days - 1)), tz),
            "days_left": max(1, days - 1),
        },
    ]
    subject, body = render_reminder_email(
        display_name="Vandal User", keys=sample, tier=tier, base_url=base_url, tz=tz
    )
    subject = f"[TEST] {subject}"
    body = (
        '<p style="background:#fff3cd;border:1px solid #ffe69c;padding:8px 12px;'
        'border-radius:4px;color:#664d03;"><strong>This is a test</strong> of the API key '
        'expiry reminder email. The keys below are fabricated samples.</p>' + body
    )
    sent = await email_service.send_notification_email(smtp, [recipient], subject, body, base_url=base_url)
    return "" if sent else "Send failed — check SMTP settings and logs"
