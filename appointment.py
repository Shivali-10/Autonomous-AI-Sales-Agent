"""Simulated CRM appointment parsing, validation, and SQLite booking."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from database import create_appointment_record, get_appointment_by_slot


BUSINESS_START_HOUR = 9
BUSINESS_END_HOUR = 19  # 7:00 PM


@dataclass
class BookingResult:
    success: bool
    message: str
    confirmation_code: str = ""
    appointment_date: str = ""
    appointment_time: str = ""
    appointment_id: Optional[int] = None
    is_simulated: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "confirmation_code": self.confirmation_code,
            "appointment_date": self.appointment_date,
            "appointment_time": self.appointment_time,
            "appointment_id": self.appointment_id,
            "is_simulated": self.is_simulated,
        }


def parse_date_time(date_str: Optional[str], time_str: Optional[str], raw_text: str = "") -> Tuple[Optional[str], Optional[str]]:
    """Normalize explicit or natural-language date and time strings to YYYY-MM-DD and HH:MM."""
    today = datetime.now().date()
    parsed_date: Optional[str] = None
    parsed_time: Optional[str] = None

    combined = f"{date_str or ''} {time_str or ''} {raw_text or ''}".lower()

    # 1. Explicit YYYY-MM-DD
    if date_str:
        m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", date_str.strip())
        if m:
            try:
                dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
                parsed_date = dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
        else:
            for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%b %d %Y", "%B %d %Y"):
                try:
                    parsed_date = datetime.strptime(date_str.strip(), fmt).date().strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue

    # Natural language date fallback
    if not parsed_date:
        if "day after tomorrow" in combined:
            parsed_date = (today + timedelta(days=2)).strftime("%Y-%m-%d")
        elif "tomorrow" in combined:
            parsed_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        elif "today" in combined:
            parsed_date = today.strftime("%Y-%m-%d")
        else:
            m_iso = re.search(r"(\d{4}-\d{2}-\d{2})", combined)
            if m_iso:
                parsed_date = m_iso.group(1)
            else:
                weekdays = {
                    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                    "friday": 4, "saturday": 5, "sunday": 6,
                }
                for wd_name, wd_num in weekdays.items():
                    if wd_name in combined:
                        days_ahead = (wd_num - today.weekday()) % 7
                        if days_ahead == 0:
                            days_ahead = 7
                        parsed_date = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
                        break

    # 2. Time extraction
    time_source = (time_str or "") + " " + (raw_text or "")
    m_ampm = re.search(r"\b(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*(am|pm)\b", time_source.lower())
    if m_ampm:
        hour = int(m_ampm.group(1))
        minute = int(m_ampm.group(2) or 0)
        meridiem = m_ampm.group(3)
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        parsed_time = f"{hour:02d}:{minute:02d}"
    else:
        m_24 = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", time_source)
        if m_24:
            parsed_time = f"{int(m_24.group(1)):02d}:{int(m_24.group(2)):02d}"

    return parsed_date, parsed_time


def validate_appointment(
    customer_name: str,
    customer_contact: str,
    appointment_date: str,
    appointment_time: str,
    db_path: Optional[str] = None,
) -> Tuple[bool, str]:
    """Validate customer details, date/time format, business hours, and slot availability."""
    if not customer_name or customer_name.strip().lower() in {"", "unknown", "guest"}:
        return False, "Please provide your full name before confirming the appointment."

    if not customer_contact or not customer_contact.strip():
        return False, "Please share an email address or phone number so we can send your simulated booking confirmation."

    if not appointment_date or not appointment_time:
        return False, "Please specify both a preferred date and time (e.g., tomorrow at 3:00 PM)."

    try:
        dt = datetime.strptime(f"{appointment_date} {appointment_time}", "%Y-%m-%d %H:%M")
    except ValueError:
        return False, "Invalid date or time format. Please use YYYY-MM-DD and HH:MM (e.g., 2026-11-27 14:30)."

    if dt < datetime.now() - timedelta(minutes=5):
        return False, f"The requested slot ({appointment_date} {appointment_time}) is in the past. Please choose a future time."

    if dt.weekday() == 6:
        return False, "Our demo consultations run Monday through Saturday. Please choose a weekday or Saturday."

    if dt.hour < BUSINESS_START_HOUR or dt.hour >= BUSINESS_END_HOUR:
        return False, (
            f"Demo slots are available between {BUSINESS_START_HOUR:02d}:00 and {BUSINESS_END_HOUR:02d}:00. "
            "Please pick a time within business hours."
        )

    existing = get_appointment_by_slot(appointment_date, appointment_time, db_path=db_path)
    if existing:
        return False, (
            f"The slot {appointment_date} at {appointment_time} is already reserved. "
            "Please select another time (e.g., 1 hour later)."
        )

    return True, "Slot is valid and available."


def book_appointment(
    session_id: str,
    lead_id: Optional[int],
    customer_name: str,
    customer_contact: str,
    appointment_date: str,
    appointment_time: str,
    purpose: str = "Product Demo & Consultation",
    notes: str = "",
    db_path: Optional[str] = None,
) -> BookingResult:
    """Validate and persist a simulated CRM appointment in SQLite."""
    norm_date, norm_time = parse_date_time(appointment_date, appointment_time)
    norm_date = norm_date or appointment_date
    norm_time = norm_time or appointment_time

    is_valid, reason = validate_appointment(
        customer_name=customer_name,
        customer_contact=customer_contact,
        appointment_date=norm_date,
        appointment_time=norm_time,
        db_path=db_path,
    )
    if not is_valid:
        return BookingResult(success=False, message=reason)

    code = f"FTX-{norm_date.replace('-', '')}-{uuid.uuid4().hex[:4].upper()}"
    appt_id = create_appointment_record(
        session_id=session_id,
        lead_id=lead_id,
        customer_name=customer_name.strip(),
        customer_contact=customer_contact.strip(),
        appointment_date=norm_date,
        appointment_time=norm_time,
        purpose=purpose,
        notes=notes or "Booked via Autonomous AI Sales Voice Agent (Simulated local CRM booking)",
        confirmation_code=code,
        status="Confirmed (Simulated)",
        db_path=db_path,
    )

    return BookingResult(
        success=True,
        message=(
            f"Simulated appointment confirmed for {customer_name.strip()} on {norm_date} at {norm_time}. "
            f"Reference ID: {code} (stored in local SQLite CRM)."
        ),
        confirmation_code=code,
        appointment_date=norm_date,
        appointment_time=norm_time,
        appointment_id=appt_id,
        is_simulated=True,
    )
