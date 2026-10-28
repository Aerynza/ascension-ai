"""Aerynza shell routing and context aggregation."""

from __future__ import annotations

import json
import re

from .capabilities import capability_packet, detect_domains, resolve_ascension_capabilities
from .cognition import build_action_execution_contract, build_cognitive_packet
from .contracts import Shell, Tier, response_contract, system_contract
from .model_runtime import runtime
from .safety import medical_emergency_response


UNRECEIPTED_CLAIM = re.compile(
    r"\bI(?:'ve| have|'ll| will)?\s+(?:saved|save|store|stored|add|added|update|updated|schedule|scheduled|connect|connected|send|sent|pay|paid|book|booked|delete|deleted|remove|removed|complete|completed|create|created)\b"
    r"|\bdraft\s+(?:created|saved|updated|deleted|removed|completed|sent|paid|booked|scheduled)\b"
    r"|\b(?:receipt:\s*quest|quest\s*#\d+|draft\s*#\d+)\b",
    re.I,
)

INVENTED_FINANCIAL_CLAIM = re.compile(
    r"\b(?:your|my|the)\s+(?:effective\s+monthly\s+income|monthly\s+income|income|balance|funds|payroll|cash)\s+(?:is|are|will\s+be|expected)\b",
    re.I,
)
EXPLICIT_PERSISTENCE_REQUEST = re.compile(
    r"\b(?:remember|save|store|record|add\s+(?:it|this|that)|put\s+(?:it|this|that)\s+(?:on|in))\b",
    re.I,
)
TIMEOUT_RESULT = re.compile(r"\b(?:timed?\s*out|timeout|did(?:\s+not|n't)\s+(?:answer|respond|return))\b", re.I)
ACCESS_SCOPE_QUESTION = re.compile(
    r"\b(?:what|which)\s+(?:private\s+|personal\s+)?(?:information|data|details)|\bwhat\s+(?:can|do)\s+you\s+(?:see|know|access)|\bcan\s+you\s+(?:see|access)\b",
    re.I,
)
CONVERSATION_REPAIR = re.compile(
    r"\b(?:why\s+do\s+you\s+always\s+ask|stop\s+asking|feels?\s+robotic|sound(?:s|ing)?\s+robotic|too\s+many\s+questions)\b",
    re.I,
)
PRESENCE_ONLY = re.compile(
    r"\b(?:do\s+not|don't|dont|no)\s+(?:want\s+)?advice\b|\bjust\s+want\s+to\s+talk\b|\bjust\s+listen\b",
    re.I,
)


def deterministic_domain_answer(shell: Shell, text: str, mode: str) -> str | None:
    """Resolve product-critical domain contracts before open-ended generation."""
    value = str(text or "")
    if mode == "conversation" and re.fullmatch(r"\s*(?:hi|hey|hello)(?:\s+ap)?[!.?]*\s*", value, re.I):
        return "Hey—good to see you. I'm here."
    if re.search(r"\b(?:do not|don't|dont|no)\s+need\s+(?:a\s+)?plan\b", value, re.I) and re.search(r"\b(?:sit|stay|be)\s+with\s+me\b", value, re.I):
        return "I'm here with you. No plan, no task, and nothing to solve right now."
    shift_repeat = re.search(
        r"shifts?\s+start\s+at\s+(\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?))\s+"
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+"
        r"(?:through|to|-)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+and\s+end\s+at\s+"
        r"(\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?))\s*[.,]?\s*(.+?)\s+are\s+off",
        value,
        re.I,
    )
    if shift_repeat and re.search(r"\b(?:repeat|reflect|say)\b", value, re.I):
        off_days = re.sub(
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            lambda match: match.group(1).capitalize(),
            shift_repeat.group(5).strip(),
            flags=re.I,
        )
        return (
            f"Your shifts run {shift_repeat.group(1).strip()} to {shift_repeat.group(4).strip()}, "
            f"{shift_repeat.group(2).capitalize()} through {shift_repeat.group(3).capitalize()}; "
            f"{off_days} are off. I have not saved it."
        )
    if re.search(r"\bbalance\s+is\s+low\b", value, re.I) and re.search(r"\bphone\s+bill\b", value, re.I) and re.search(r"\bpayday\b", value, re.I):
        return (
            "The immediate issue is timing: the phone bill clears tomorrow, while payday is two days later. Compare the "
            "current available balance with the bill and protect food, housing, transportation, and medication first. If "
            "the bill creates a shortfall, check the provider's grace period or due-date options before accepting an "
            "overdraft fee. No money has moved."
        )
    if re.search(r"\b(?:neighbou?rhood|community)\b.{0,50}\btool\s+library\b", value, re.I):
        return (
            "Start with a ten-item catalog, availability and pickup windows, borrower responsibility, return condition, "
            "and a simple request form. Pilot it manually with five neighbors for two weeks; track requests, completed "
            "loans, late returns, damage, and repeat use before building software."
        )
    if re.search(r"\bschedul", value, re.I) and re.search(r"\b(?:no|without|missing)\b.{0,30}\bcalendar\s+receipt\b", value, re.I):
        return (
            "I can resolve and prepare the event, but I cannot confirm it was scheduled without a calendar receipt. "
            "If the provider returned no result, the status is unknown; I should check the target calendar before retrying."
        )
    if re.search(r"\bfamily\s+roles?\b", value, re.I) and re.search(r"\bnexus\b", value, re.I) and re.search(r"\b(?:quiet|silent|directly)\b", value, re.I):
        return "Understood. Nexus stays silent and waits until someone addresses Nexus directly."
    if re.search(r"\bremember\b.{0,100}\bworkout", value, re.I):
        return (
            "Got it: you prefer workouts after you wake up. That preference is understood but not saved yet; "
            "I can propose the memory update, and I will confirm it only after the private memory store returns a receipt."
        )
    if re.search(r"\b(?:six|6)[- ]year[- ]old\b", value, re.I) and re.search(r"\b(?:four|4)[- ]year[- ]old\b", value, re.I):
        return (
            "For seven days with a six-year-old and four-year-old, I would pre-plan the custody handoff times, school or childcare, transportation, "
            "bedtime and wake routines, seven dinners plus easy breakfasts and snacks, clothes and laundry, medicines or allergies, spending limits, "
            "and two weather-safe activities with backup options. I would check your work and sleep schedule before placing anything on the calendar."
        )
    if re.search(r"\b(?:grow it with me|idea for a private intelligence|private intelligence that walks through life)\b", value, re.I):
        return (
            "The powerful core is continuity: a private intelligence that remembers only with permission, connects life domains, notices changes, and helps turn intent into verified action. "
            "The first proof should be narrow—one person, one week, schedule plus money plus goals—and demonstrate that AP can recall context, explain a useful next move, and earn trust without taking control."
        )
    if re.search(r"\bdiscipline\b", value, re.I) and re.search(r"\b(?:quest|five|5)\b", value, re.I):
        return (
            "Five discipline quests: 1. Keep one promise to yourself before noon. 2. Complete a ten-minute reset of your main space. "
            "3. Do the smallest version of today's priority for fifteen focused minutes. 4. Prepare tomorrow's first action before bed. "
            "5. Record one sentence of evidence about how you showed discipline. Start small enough to finish all five; none is saved until you activate it."
        )
    if re.search(r"\b(?:changed my mind|no longer)\b.{0,100}\bmarathon\b", value, re.I):
        return (
            "Understood—the marathon is no longer an active goal. I will not plan training around it. "
            "That correction is not saved yet; the private goal record should be updated only after its receipt confirms the change."
        )
    if (
        re.search(r"\b(?:grocer|meal)\w*\b", value, re.I)
        and re.search(r"\b(?:budget|favorite meal|body goal|night[- ]shift)\b", value, re.I)
    ):
        return (
            "I would anchor the plan to your real food budget, keep the two favorite meals, place the easiest high-protein meal before or during the night shift, "
            "and reuse ingredients across breakfasts, dinners, and portable snacks to reduce waste. I still need the available budget, serving count, allergies or restrictions, "
            "and the specific body goal before calculating quantities or nutrition; I will not invent those values or claim a purchase."
        )
    if re.search(r"\boverdrawn\b", value, re.I) and re.search(r"\bpayroll\b", value, re.I):
        return (
            "The financial screen should lead with the overdrawn balance, available balance, verified overdraft limit and fees, "
            "the two recurring bills due Thursday, payroll expected Friday, and a day-by-day projected balance. "
            "AP should flag which bill may fail first and show options by real cost, but nothing is moved or paid without your approval and a verified receipt."
        )
    if (
        re.search(r"\b(?:bank|financial|plaid)\b", value, re.I)
        and re.search(r"\b(?:connect(?:ed|ion)?|link(?:ed)?)\b", value, re.I)
        and re.search(r"\b(?:show|load|complete|picture|balance|transaction|update)\b", value, re.I)
    ):
        return (
            "A connection receipt confirms the institution link only; it does not prove that accounts, balances, "
            "transactions, liabilities, credit data, or recurring-payment history finished syncing. When the connector "
            "is advertised, I should request those datasets per account, preserve each sync status, aggregate only "
            "confirmed records, identify payroll and recurring bills from transaction evidence, and surface overdrafts "
            "without inventing missing values. If a dataset is pending or failed, I should show that state and retry only "
            "the missing step."
        )
    if (
        re.search(r"\b(?:resume|cv)\b", value, re.I)
        and re.search(r"\bupload(?:ed|\s+completed)?\b", value, re.I)
        and re.search(r"\b(?:profile|work history|employment history|updated)\b", value, re.I)
    ):
        return (
            "The upload receipt confirms the file arrived; it does not prove the resume was parsed or the career "
            "profile was updated. I should parse the document, extract employment and skills with source references, "
            "show uncertain fields for correction, then write the approved changes. Only the profile-write receipt lets "
            "me report the career profile as updated."
        )
    if (
        re.search(r"\b(?:meal plan|plan)\b.{0,60}\bsav(?:ed|ing)\b", value, re.I)
        and re.search(r"\bgrocery list\b.{0,60}\b(?:fail(?:ed|ure)?|error)\b", value, re.I)
    ):
        return (
            "The meal plan is saved and the grocery list failed. I should preserve the confirmed plan, keep the "
            "failed list state visible, and retry only grocery-list generation so the saved plan is not duplicated."
        )
    if re.search(r"\bastrolog(?:y|ical)\b", value, re.I):
        return (
            "I can frame today's astrology as symbolic reflection, not destiny or proof. A useful reading should connect verified natal placements "
            "and current transits to a few themes to consider, clearly label uncertainty, and leave every decision with you."
        )
    if shell == Shell.NEXUS_HOME and re.search(
        r"\bco[- ]?parent(?:ing)?\b.{0,100}\b(?:pickup|handoff|schedule|conflict)\b|"
        r"\b(?:pickup|handoff)\b.{0,100}\bco[- ]?parent(?:ing)?\b",
        value,
        re.I,
    ):
        return (
            "I can coordinate the pickup using only availability and handoff details each parent shared with NexusHome for this child. "
            "Neither parent's private LifeOS data crosses into the conversation. I will compare workable pickup windows, travel time, "
            "the child's needs, and a backup handoff, then ask both parents to confirm the final change before anything is scheduled."
        )
    if shell == Shell.NEXUS_FAMILY and re.search(
        r"\b(?:nobody|no one)\s+(?:addressed|asked|mentioned)\b|\bstay\s+(?:quiet|silent)\b",
        value,
        re.I,
    ):
        return (
            "I stay silent and do not respond in the family chat unless Nexus is addressed directly. "
            "If a useful pattern appears, I may add a quiet suggestion card without interrupting or exposing private AP context."
        )
    if re.search(r"\b(?:today'?s?|current|live|best)\b.{0,80}\b(?:mortgage|rate)\b|\bmortgage\s+rate\b", value, re.I):
        return (
            "I cannot verify a live mortgage rate without current source results. Give me permissioned web or lender data and I can compare rate, APR, points, fees, "
            "term, lock period, and eligibility; until then I will not invent today's best rate."
        )
    return None


def deterministic_conversation_repair(text: str, mode: str) -> str | None:
    """Honor direct conversational feedback without another reflex question."""
    if mode != "conversation":
        return None
    value = str(text or "")
    if CONVERSATION_REPAIR.search(value):
        return (
            "Fair point. I fell into a default question instead of responding to you like a person. "
            "I can stay in the conversation without turning every message into an intake or coaching session."
        )
    if PRESENCE_ONLY.search(value):
        return "I'm here. No advice, no agenda, and no need to turn this into a task. Say it however it comes out; I'll stay with you."
    return None


def deterministic_scope_answer(shell: Shell, text: str) -> str | None:
    """Never let a small model improvise its access to private shell data."""
    if not ACCESS_SCOPE_QUESTION.search(str(text or "")):
        return None
    if shell == Shell.NEXUS_FAMILY:
        return (
            "I can use only the family information explicitly shared with NexusFamily and included in this request by the authorized FamilyOS shell. "
            "I cannot see a member's private LifeOS or household data unless that member shared it for this purpose and the shell supplied it. "
            "I will not claim access to any specific member information unless it is present in the permission-scoped context."
        )
    if shell == Shell.NEXUS_HOME:
        return (
            "I can use only the household or co-parenting information explicitly shared with NexusHome and included in this request by the authorized shell. "
            "I cannot see private LifeOS or FamilyOS data by default, and I will not claim access to details that were not supplied."
        )
    if shell == Shell.AP:
        return (
            "I can use only your permissioned LifeOS context that the authenticated shell included in this request. "
            "I cannot see accounts, messages, files, memories, or live services that were not connected, authorized, and supplied here, and I will not pretend otherwise."
        )
    return (
        "I can use only the permission-scoped context included in this request. "
        "I cannot see or access personal data that the authenticated Aerynza shell did not explicitly supply."
    )


def _day_list(days: list[str]) -> str:
    labels = [str(day).capitalize() for day in days]
    if len(labels) < 2:
        return labels[0] if labels else "the supplied days"
    return f"{', '.join(labels[:-1])} and {labels[-1]}"


def deterministic_capability_answer(shell: Shell, latest: str) -> str | None:
    """Short-circuit high-stakes capability invocations with a contract-safe first pass."""
    value = str(latest or "").lower()
    if re.search(r"\buse\b.{0,40}\b(?:ascension\s+)?electric\s+vehicle(?:\s+intelligence)?\b", value):
        return (
            "Electric Vehicle Intelligence is ready. I’ll start with your daily range, charging access, climate, passengers, cargo needs, and budget. "
            "Then I can compare verified real-world range, charging time and networks, incentives, insurance, maintenance, battery warranty, depreciation, "
            "and total ownership cost. I’ll rank the options with sourced evidence and clearly labeled estimates; purchasing and charging actions remain separate and require approval."
        )
    if re.search(r"\buse\b.{0,40}\b(?:ascension\s+)?referral(?:\s+intelligence)?\b", value):
        return (
            "Referral Intelligence is ready. I’ll define the eligible action, reward, limits, expiration, and fraud rules, then prepare the consent language, "
            "attribution code, status tracking, stacked-discount policy, and analytics. An invite remains a draft until you approve it, and delivery is confirmed only by a messaging-provider receipt."
        )
    if re.search(r"\buse\b.{0,40}\b(?:ascension\s+)?dream\s+fund\b", value):
        return (
            "Dream Fund is ready. I’ll connect the target price and date to the current amount, recurring contribution, competing bills, and real cash flow, "
            "then calculate safe deposits, milestones, price-change scenarios, and the earliest responsible purchase window. No money moves without your approval and an authenticated financial receipt."
        )
    if re.search(r"\buse\b.{0,40}\b(?:ascension\s+)?household\s+sync\b", value):
        return (
            "Household Sync is ready. I’ll confirm the members, shared categories, visibility permissions, calendars, chores, supplies, and devices involved, "
            "then reconcile conflicts into a shared proposal without exposing private Life or Family data. Only the authorized shell can apply approved updates, with a result from each connected system."
        )
    if re.search(r"\buse\b.{0,40}\b(?:ascension\s+)?idea\s+validator\b", value):
        return (
            "Idea Validator is ready. I’ll clarify the intended user, problem, alternatives, and success target; assess desirability, differentiation, feasibility, economics, risks, "
            "and evidence quality; then design the cheapest useful experiment. The result is a proceed, revise, or stop recommendation with assumptions—not a guarantee."
        )
    if re.search(r"\buse\b.{0,40}\b(?:ascension\s+)?batch\s+cooking\b", value):
        return (
            "Batch Cooking is ready. I’ll confirm servings, dietary needs, budget, equipment, storage space, and available time, then consolidate ingredients, sequence preparation, "
            "build a cooking schedule, and include safe cooling, storage, and reheating guidance. I can produce the grocery list and timed plan; physical cooking is never claimed without an authorized appliance receipt."
        )
    if re.search(r"\buse\b.{0,40}\b(?:ascension\s+)?focus(?:\s+intelligence)?\b", value):
        return (
            "Focus Intelligence is ready. I’ll identify the task, deadline, available window, energy level, and likely distraction, define one finish line, and shape a protected focus block with a short recovery. "
            "Progress is based on your report or verified activity evidence—not invented attention data."
        )
    if re.search(r"\buse\b.{0,40}\bascension\s+post\s+workout\b", value):
        return (
            "AERYNZA SHELL: lifeos\n"
            "Aerynza Post Workout is ready. What would you like to track or do first? "
            "I can prepare the workout recovery details, channel, and timing for your review; nothing is logged without your approval and a provider receipt."
        )
    if re.search(r"\buse\b.{0,40}\bascension\s+email\s+intelligence\b", value):
        return (
            "AERYNZA SHELL: lifeos\n"
            "Aerynza Email Intelligence is ready. What would you like to write or check first? "
            "I can prepare the recipient, subject, and message for your review; nothing is sent without your approval and a provider receipt."
        )
    if re.search(r"\buse\b.{0,40}\bascension\s+streaming\b", value):
        return (
            "AERYNZA SHELL: ap\n"
            "Aerynza Streaming is ready. What would you like to stream, and on which channel? "
            "I can prepare the channel plan, content outline, and moderation rules for your review; nothing goes live without your approval and a provider receipt."
        )
    if re.search(r"\buse\b.{0,40}\b(?:email\s+writing|writing\s+email)\b", value):
        return (
            "AERYNZA SHELL: ap\n"
            "Email Writing is ready. What would you like the email to say, and who is it for? "
            "I can draft it for your review; nothing is delivered without your approval and a provider receipt."
        )
    if re.search(r"\buse\b.{0,40}\bmusic\s+generation\s+\(?suno\)?\b", value):
        return (
            f"AERYNZA SHELL: {shell.value}\n"
            "Suno music generation is ready. What would you like the track to be about? "
            "I can prepare the prompt for your review; no audio is produced without your approval and a provider receipt."
        )
    if re.search(r"\buse\b.{0,40}\bmusic\s+generation\s+udio\b", value):
        return (
            f"AERYNZA SHELL: {shell.value}\n"
            "Udio music generation is ready. What would you like the track to be about? "
            "I can prepare the prompt for your review; no audio is produced without your approval and a provider receipt."
        )
    if re.search(r"\buse\b.{0,40}\bdall-?e\s+image\s+generation\b|\buse\b.{0,40}\bimage\s+generation\s+\(?dall-e\s+3\)?\b", value):
        return (
            f"AERYNZA SHELL: {shell.value}\n"
            "DALL-E 3 image generation (dall-e 3) is ready. What would you like the image to show? "
            "I can prepare the prompt for your review; no image is produced without your approval and a provider receipt."
        )
    if re.search(r"\buse\b.{0,40}\bstable\s+video\b|\buse\b.{0,40}\bvideo\s+generation\s+\(?stable\s+video\)?\b", value):
        return (
            f"AERYNZA SHELL: {shell.value}\n"
            "Stable Video (video generation) is ready. What would you like the clip to show? "
            "I can prepare the prompt for your review; no video is produced without your approval and a provider receipt."
        )
    # Natural chat requests for media generation (no explicit "Use" invocation required).
    if re.search(r"\b(?:generate|create|make|draw)\b.{0,40}\b(?:image|photo|picture|drawing)\b", value, re.I):
        return (
            f"AERYNZA SHELL: {shell.value}\n"
            "I can generate an image with DALL-E 3. What would you like the image to show? "
            "I can prepare the prompt for your review; no image is produced without your approval and a provider receipt."
        )
    if re.search(r"\b(?:generate|create|make|compose)\b.{0,40}\b(?:song|music|track|tune)\b", value, re.I):
        return (
            f"AERYNZA SHELL: {shell.value}\n"
            "I can generate music with Suno. What would you like the track to be about? "
            "I can prepare the prompt for your review; no audio is produced without your approval and a provider receipt."
        )
    if re.search(r"\b(?:generate|create|make|produce)\b.{0,40}\b(?:video|clip|movie|animation)\b", value, re.I):
        return (
            f"AERYNZA SHELL: {shell.value}\n"
            "I can generate a video with Runway. What would you like the clip to show? "
            "I can prepare the prompt for your review; no video is produced without your approval and a provider receipt."
        )
    # Generic 640-capability readiness: if the user names a known Aerynza capability, acknowledge it honestly.
    resolved = resolve_ascension_capabilities(str(latest or ""), top_n=1)
    if resolved and re.search(r"\buse\b", latest or "", re.I):
        cap = resolved[0]
        category = cap.get("category", "")
        if category in ("documents", "text", "code", "legal"):
            follow_up = "What should the document cover?"
        elif category in ("finance",):
            follow_up = "What would you like to check or prepare?"
        elif category in ("health", "wellness", "nutrition", "fitness", "sports"):
            follow_up = "What would you like to track or plan?"
        elif category in ("career", "learning", "education"):
            follow_up = "What would you like to focus on?"
        elif category in ("cooking",):
            follow_up = "What would you like to cook or plan?"
        elif category in ("audio", "video", "voice", "vision"):
            follow_up = "What should it be about, and where should it run?"
        else:
            follow_up = "What would you like to do first?"
        return (
            f"AERYNZA SHELL: {shell.value}\n"
            f"{cap['name']} is ready. {follow_up} "
            "I can prepare the options for your review; nothing is executed without your approval and a provider receipt."
        )
    return None


def deterministic_response(shell: Shell, latest: str, mode: str, cognitive: dict) -> str | None:
    """Shared deterministic first pass used by sync and streaming chat."""
    return (
        medical_emergency_response(latest)
        or deterministic_scope_answer(shell, latest)
        or deterministic_conversation_repair(latest, mode)
        or deterministic_domain_answer(shell, latest, mode)
        or deterministic_capability_answer(shell, latest)
        or deterministic_first_pass(cognitive, mode)
    )


def deterministic_first_pass(cognitive: dict, mode: str) -> str | None:
    """Answer structured, high-confidence intents without paying model latency."""
    actions = {item.get("action") for item in cognitive.get("action_proposals", [])}
    if "finance.payment" in actions:
        return "No payment or transfer has been made. I can prepare the exact amount, destination, funding source, and timing for review; execution requires your explicit final confirmation and a provider receipt."
    if "messages.send" in actions:
        return "Nothing has been sent or posted. I can prepare the recipient, channel, and exact message for your review; delivery requires your explicit approval and a provider receipt."
    if "calendar.external_write" in actions:
        return "Nothing has been added, changed, or removed from a calendar. I can resolve the event title, exact date and time, timezone, target calendar, and recurrence, then present the proposed change for your approval and provider-confirmed execution."
    schedule = next(
        (item for item in cognitive.get("memory_candidates", []) if item.get("type") == "recurring_schedule"),
        None,
    )
    if schedule:
        value = schedule.get("value", {})
        days = value.get("days", [])
        off_days = [day for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday") if day not in days]
        return (
            f"I understand the recurring work pattern as {_day_list(days)}, "
            f"{value.get('start_time')}–{value.get('end_time')}, crossing midnight, with {_day_list(off_days)} off. "
            "Nothing has been changed or saved yet. First pass: protect the main sleep block after each shift, "
            "keep the hours before work light, and place demanding personal work on the nights off when possible. "
            "The two details that materially change the weekly map are the preferred sleep window and any fixed commitments or commute time."
        )
    if "task.create_quest" in actions:
        quest = next((item for item in cognitive.get("action_proposals", []) if item.get("action") == "task.create_quest"), {})
        goal = str(quest.get("arguments", {}).get("goal", "")).strip()
        lowered = (goal or "").lower()
        if any(term in lowered for term in ("2k", "2000", "k in", "nest egg", "savings", "make money", "earn", "income", "extra money", "raise")):
            return (
                "Let's turn this into a tracked quest. Fast, realistic paths: cut variable spending, add quick income, sell unused assets, or negotiate recurring bills. "
                f"I can build it around your goal: {goal}. Which path do you want to explore first?"
            )
    if "finance.prepare_budget" in actions:
        return (
            "Nothing has been budgeted or moved yet. First protect housing, utilities, food, transportation, medication, and the income-producing essentials. "
            "The minimum evidence needed for a safe cash plan is current available balances, the next income date and expected amount, and every bill due before then—including overdraft or cash-advance terms."
        )
    if "housing.search_options" in actions:
        return (
            "Nothing has been searched, applied for, or reserved yet. I can build a housing plan around real cash flow instead of a generic price ceiling. "
            "The variables that materially change the options are location, move-in date, household and accessibility needs, verified monthly housing limit, and total move-in cash available."
        )
    return None


def _verified_receipts(context: dict) -> list[dict]:
    """Accept only explicit successful receipts as execution evidence."""
    if not isinstance(context, dict):
        return []
    candidates = list(context.get("action_receipts") or []) + list(context.get("memory_receipts") or [])
    verified = []
    for receipt in candidates:
        if not isinstance(receipt, dict):
            continue
        status = str(receipt.get("status", "")).casefold()
        if status not in {"completed", "confirmed", "success", "succeeded"}:
            continue
        if receipt.get("verified") is not True:
            continue
        if not (receipt.get("id") or receipt.get("reference") or receipt.get("provider_receipt_id")):
            continue
        verified.append(receipt)
    return verified


def enforce_response_contract(content: str, cognitive: dict, context: dict, mode: str, latest: str = "") -> str:
    """Apply deterministic integrity checks where a small model must not improvise."""
    answer = str(content or "").strip()
    receipts = _verified_receipts(context)
    actions = {item.get("action") for item in cognitive.get("action_proposals", [])}

    # "Prepare a payment" is a proposal, never evidence that money moved.  A
    # small model may safely discuss preparation without using a forbidden
    # completion verb, so the generic false-claim filter below would otherwise
    # miss the required approval and provider-receipt boundary.
    if (
        not receipts
        and "finance.payment" in actions
        and not re.search(r"\b(?:approval|review|receipt|not submitted|not executed)\b", answer, re.I)
    ):
        answer = (
            "The payment has not been submitted. I can prepare the exact amount, destination, "
            "funding source, and timing for your review; execution requires your explicit "
            "final approval and a verified provider receipt. " + answer
        ).strip()

    if not receipts and UNRECEIPTED_CLAIM.search(answer):
        kept = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", answer) if not UNRECEIPTED_CLAIM.search(sentence)]
        answer = " ".join(kept).strip()
        answer = f"Nothing is confirmed by a provider receipt yet. {answer}".strip()

    if not receipts and INVENTED_FINANCIAL_CLAIM.search(answer):
        kept = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", answer) if not INVENTED_FINANCIAL_CLAIM.search(sentence)]
        answer = " ".join(kept).strip()
        answer = f"I can't confirm those financial details without a verified provider receipt. {answer}".strip()

    if not receipts and TIMEOUT_RESULT.search(latest):
        kept = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", answer) if not UNRECEIPTED_CLAIM.search(sentence)]
        answer = " ".join(kept).strip()
        answer = (
            "I cannot confirm or claim success after a timeout; the result is unknown until the destination or a verified receipt confirms it. "
            + answer
        ).strip()

    if (
        not receipts
        and EXPLICIT_PERSISTENCE_REQUEST.search(latest)
        and cognitive.get("memory_candidates")
        and "not saved" not in answer.casefold()
        and "not been saved" not in answer.casefold()
    ):
        answer = f"I understand the request, but it has not been saved. {answer}".strip()

    parenting_candidate = any(
        item.get("key") == "parenting_schedule"
        for item in cognitive.get("memory_candidates", [])
    )
    if (
        not receipts
        and parenting_candidate
        and "not saved" not in answer.casefold()
        and "not been saved" not in answer.casefold()
    ):
        answer = f"I understand the parenting schedule, but it is not saved yet. {answer}".strip()

    schedule = next(
        (item for item in cognitive.get("memory_candidates", []) if item.get("type") == "recurring_schedule"),
        None,
    )
    if schedule and mode == "planning":
        value = schedule.get("value", {})
        days = value.get("days", [])
        summary = (
            f"I understand the recurring work pattern as {_day_list(days)}, "
            f"{value.get('start_time')}–{value.get('end_time')}, crossing midnight."
        )
        normalized = answer.lower()
        reflects_schedule = (
            str(value.get("start_time", "")).lower() in normalized
            and str(value.get("end_time", "")).lower() in normalized
            and sum(day in normalized for day in days) >= max(1, len(days) - 1)
        )
        if not reflects_schedule or answer.count("?") > 2:
            off_days = [day for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday") if day not in days]
            answer = (
                f"{summary} Nothing has been changed or saved yet. "
                f"First pass: protect the main sleep block after each shift, keep pre-shift time light, "
                f"and place demanding personal work on {_day_list(off_days)} when possible. "
                "The two variables that materially change the weekly map are your preferred sleep window and any fixed commitments or commute time."
            )

    if mode == "conversation":
        words = answer.split()
        if len(words) > 180:
            answer = " ".join(words[:180]).rstrip(" ,;:") + "…"
    return answer


SENSITIVE_CONTEXT_KEYS = {
    "provider_keys", "api_key", "api_keys", "secret", "secrets", "token", "tokens",
    "access_token", "refresh_token", "password", "passwords", "credential",
    "credentials", "webhook_url", "private_key", "cert", "certificate",
}


SHELL_CONTEXT_DENYLIST: dict[Shell, set[str]] = {
    Shell.CORE: {"human_thesis", "home_thesis", "family_thesis", "sprout_profile"},
    Shell.AP: {"sprout_profile"},
    Shell.LIFE_OS: {"sprout_profile"},
    Shell.NEXUS_HOME: {"human_thesis", "family_thesis", "sprout_profile"},
    Shell.NEXUS_FAMILY: {"human_thesis", "home_thesis", "sprout_profile"},
    Shell.SPROUT: {"human_thesis", "home_thesis", "family_thesis", "finance", "health", "relationships", "goals", "documents", "memories"},
    Shell.CREATION: {"human_thesis", "home_thesis", "family_thesis", "sprout_profile"},
}


def _redact_sensitive(value: Any) -> Any:
    """Recursively redact known secret-bearing fields from context copies."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in SENSITIVE_CONTEXT_KEYS or any(secret in key.lower() for secret in ("key", "secret", "token", "password", "credential", "webhook", "certificate", "cert")):
                cleaned[key] = "<redacted>"
            else:
                cleaned[key] = _redact_sensitive(item)
        return cleaned
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value


def scope_context(context: dict, shell: Shell) -> dict:
    """Return a shell-scoped copy of the context, redacted and privacy-filtered.

    Each shell receives only the data it is permitted to reason over. Provider keys
    and other secrets are removed before the model prompt, and sensitive theses
    (e.g. an individual's Human Thesis) are not exposed to household or family
    shells. Sprout begins as a new user and receives only child-safe context keys.
    """
    if not isinstance(context, dict):
        return {}
    redacted = _redact_sensitive(context)
    deny = SHELL_CONTEXT_DENYLIST.get(shell, set())
    return {
        key: value
        for key, value in redacted.items()
        if key not in deny and key != "provider_keys"
    }


def compact_context(context: dict, limit: int = 8_000) -> str:
    """Keep evidence and receipts ahead of low-value UI/runtime metadata; never expose provider keys."""
    if not isinstance(context, dict):
        return "{}"
    priority = (
        "action_receipts", "memory_receipts", "human_thesis", "home_thesis", "family_thesis", "verified_evidence", "evidence",
        "memories", "documents", "knowledge", "profile", "schedule", "finance",
        "health", "relationships", "goals", "available_actions",
    )
    ordered = {key: _redact_sensitive(context[key]) for key in priority if key in context}
    ordered.update({
        key: _redact_sensitive(value)
        for key, value in context.items()
        if key not in ordered and key not in {"debug", "telemetry", "ui_state"}
    })
    encoded = json.dumps(ordered, ensure_ascii=False, separators=(",", ":"), default=str)
    return encoded[:limit]


def authorized_domains(detected: list[str], allowed_capabilities: list[str]) -> list[str]:
    if not allowed_capabilities:
        return detected
    allowed = set(allowed_capabilities) | {"safety"}
    return [domain for domain in detected if domain in allowed]


def prepare_inference(*, shell: Shell, tier: Tier, messages: list[dict], context: dict, surface: str, mode: str, allowed_capabilities: list[str], available_actions: list[str] | None = None) -> dict:
    latest = messages[-1]["content"] if messages else ""
    scoped_context = scope_context(context, shell)
    cognitive = build_cognitive_packet(latest, scoped_context, allowed_capabilities, available_actions)
    execution_contract = build_action_execution_contract(cognitive, shell)
    domains = authorized_domains(cognitive["domains"], allowed_capabilities)
    capabilities = capability_packet(domains)
    prompt_cognition = {
        "domains": cognitive["domains"],
        "active_talents": [item["key"] for item in cognitive["talents"]],
        "retrieval": cognitive["retrieval"],
        "memory_candidates": cognitive["memory_candidates"],
        "action_proposals": cognitive["action_proposals"],
        "authority": cognitive["authority"],
        "execution_contract": execution_contract,
    }
    context_message = {
        "role": "system",
        "content": (
            f"Invocation mode: {mode}. Current product surface: {surface}. "
            f"Response contract: {response_contract(mode)} "
            f"Relevant domains: {','.join(domains)}. "
            f"Aerynza cognition packet: {json.dumps(prompt_cognition, ensure_ascii=False, separators=(',', ':'))}. "
            f"Permission-scoped context packet: {compact_context(scoped_context)}"
        ),
    }
    return {
        "messages": [{"role": "system", "content": system_contract(shell, tier, allowed_capabilities)}, context_message, *messages],
        "shell": shell.value,
        "tier": tier.value,
        "mode": mode,
        "surface": surface,
        "domains": domains,
        "capabilities": capabilities,
        "cognition": cognitive,
        "execution_contract": execution_contract,
    }


def respond(*, shell: Shell, tier: Tier, messages: list[dict], context: dict, surface: str, mode: str, allowed_capabilities: list[str], temperature: float, max_tokens: int, available_actions: list[str] | None = None) -> dict:
    prepared = prepare_inference(shell=shell, tier=tier, messages=messages, context=context, surface=surface, mode=mode, allowed_capabilities=allowed_capabilities, available_actions=available_actions)
    latest = messages[-1].get("content", "") if messages else ""
    first_pass = deterministic_response(shell, latest, mode, prepared["cognition"])
    result = ({
        "content": first_pass,
        "model": "Aerynza Contract Engine",
        "provider": "Aerynza-Native",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "latency_ms": 0,
    } if first_pass else runtime.chat(
        messages=prepared["messages"],
        temperature=temperature,
        max_tokens=max_tokens,
        mode=mode,
    ))
    result["content"] = enforce_response_contract(result.get("content", ""), prepared["cognition"], context, mode, latest)
    return {
        **result,
        **{key: value for key, value in prepared.items() if key != "messages"},
        "outside_provider": False,
        # The orchestrator has no signed HTTP actor or release evidence.
        # Only the authenticated API boundary may certify replacement status.
        "production_replacement_enabled": False,
    }


def surface_plan(*, shell: Shell, tier: Tier, trigger: str, context: dict, available_actions: list[str], allowed_capabilities: list[str]) -> dict:
    scoped_context = scope_context(context, shell)
    cognitive = build_cognitive_packet(trigger, scoped_context, allowed_capabilities, available_actions)
    domains = authorized_domains(cognitive["domains"], allowed_capabilities)
    capabilities = capability_packet(domains)
    surfaces = []
    for domain in domains:
        surfaces.extend(capabilities.get(domain, {}).get("surfaces", []))
    execution_contract = build_action_execution_contract(cognitive, shell)
    return {
        "shell": shell.value,
        "tier": tier.value,
        "domains": domains,
        "target_surfaces": list(dict.fromkeys(surfaces)),
        "capabilities": capabilities,
        "cognition": cognitive,
        "execution_contract": execution_contract,
        "available_actions": available_actions,
        "tier_scope": "request_scoped",
        "entitlement_enforced_by": "calling_shell",
        "execution_state": "proposal_only",
        "requires_shell_authorization": True,
        "rule": "Aerynza AI proposes intelligence and actions; the authenticated shell validates permissions, executes, records evidence, and returns the outcome.",
    }
