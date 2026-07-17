"""Email tool."""
from __future__ import annotations

import re
import smtplib
import ssl
from email.message import EmailMessage

import config

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def send_email(args: dict) -> dict:
    recipient = str((args or {}).get("recipient", "")).strip()
    subject = str((args or {}).get("subject", "")).strip()
    body = str((args or {}).get("body", "")).strip()
    if not EMAIL_RE.match(recipient):
        return {"error": "invalid recipient email"}
    if config.EMAIL_ALLOWED_RECIPIENTS and recipient.lower() not in config.EMAIL_ALLOWED_RECIPIENTS:
        return {"error": "recipient is not in NEURAL_MAP_EMAIL_ALLOWED_RECIPIENTS"}
    if len(body) > config.MAX_EMAIL_BODY_CHARS:
        return {"error": "email body too long"}
    draft = {"recipient": recipient, "subject": subject, "body": body}
    if not config.EMAIL_SEND_ENABLED:
        return {**draft, "sent": False, "draft": True, "reason": "NEURAL_MAP_EMAIL_SEND is not enabled"}
    if not config.EMAIL_ALLOWED_RECIPIENTS:
        return {**draft, "sent": False, "error": "real email sending requires NEURAL_MAP_EMAIL_ALLOWED_RECIPIENTS"}
    if not (config.SMTP_HOST and config.SMTP_FROM):
        return {**draft, "sent": False, "error": "SMTP settings are incomplete"}
    msg = EmailMessage()
    msg["From"] = config.SMTP_FROM
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=10) as smtp:
            if config.SMTP_STARTTLS:
                smtp.starttls(context=ssl.create_default_context())
            if config.SMTP_USER:
                smtp.login(config.SMTP_USER, config.SMTP_PASSWORD)
            smtp.send_message(msg)
    except Exception as exc:
        return {**draft, "sent": False, "error": str(exc)}
    return {**draft, "sent": True}
