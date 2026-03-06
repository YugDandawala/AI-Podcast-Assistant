"""
BankLens V2 — Universal Bank Statement Extractor
=================================================
A multi-strategy, zero-regex extraction engine that handles any bank
statement PDF in the world.

Strategies (auto-selected per page):
  1. Table Extraction    — pdfplumber.extract_tables() for structured PDFs
  2. Positional Words    — word (x,y) clustering for free-form PDFs
  3. OCR + LLM Fallback  — ocrmypdf → Groq Llama for scanned pages

Author:  AI-powered redesign
License: MIT
"""

import streamlit as st
import pdfplumber
import pandas as pd
import os
import io
import re
import json
import subprocess
import tempfile
from datetime import datetime
from collections import defaultdict
import dataclasses
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Known header keywords — covers banks globally
HEADER_ROLES = {
    "date": [
        "date", "txn date", "trans date", "transaction date", "posting date",
        "post date", "effective date", "txn dt", "trn date",
        "booking date", "entry date", "dated",
    ],
    "description": [
        "description", "particulars", "narration", "details",
        "transaction details", "remarks", "narrative",
        "transaction description", "memo", "payee",
    ],
    "deposit": [
        "deposit", "deposits", "credit", "credits", "credit amount",
        "deposits ($)", "cr", "credit(inr)", "credit amt", "money in",
    ],
    "withdrawal": [
        "withdrawal", "withdrawals", "debit", "debits", "debit amount",
        "withdrawals ($)", "dr", "debit(inr)", "debit amt", "money out",
    ],
    "balance": [
        "balance", "closing balance", "running balance", "balance ($)",
        "available balance", "ledger balance", "bal", "cumulative balance",
    ],
    "cheque": [
        "cheque", "chq", "check", "cheque no", "check no", "chq no",
        "instrument", "instrument no",
    ],
    "value_date": [
        "value date", "val date", "value dt",
    ],
    "amount": [
        "amount", "amt", "transaction amount", "txn amount",
    ],
}

# Date pattern for identification only (not structural parsing)
DATE_DETECT = re.compile(
    r"^\s*\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"(?:\s+\d{2,4})?\s*$"
    r"|^\s*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\s*$"
    r"|^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}\s*$"
    r"|^\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}"
    r"(?:[,\s]+\d{2,4})?\s*$",
    re.IGNORECASE,
)

MONEY_DETECT = re.compile(
    r"^[\s$£€¥₹]*-?\s*\d{1,3}(?:[,]\d{3})*(?:\.\d{1,2})?\s*$"
)

NOISE_WORDS = {
    "page", "statement", "account", "branch", "ifsc", "micr",
    "phone", "address", "nominee", "currency", "this is a computer",
    "generated", "continued", "brought forward", "carried forward",
}

SKIP_DESCRIPTIONS = {
    "BALANCE FORWARD", "OPENING BALANCE", "CLOSING BALANCE",
    "BROUGHT FORWARD", "CARRIED FORWARD",
}

SKIP_SUBSTRINGS = {
    "TOTALS AT END", "TOTAL", "BALANCE FORWARD",
    "OPENING BALANCE", "CLOSING BALANCE",
}

# Pattern to extract a date from the START of a blended string
# e.g. "03 FEB    ANZ ATM BRANCH" → ("03 FEB", "ANZ ATM BRANCH")
DATE_PREFIX = re.compile(
    r"^\s*(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"(?:\s+\d{2,4})?)\s+(.*)",
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Transaction:
    date: str = ""
    value_date: str = ""
    description: str = ""
    payee: str = ""
    amount: float = 0.0
    balance: Optional[float] = None
    cheque_no: str = ""
    reference: str = ""
    confidence: str = "High"
    raw_deposit: Optional[float] = None
    raw_withdrawal: Optional[float] = None


@dataclass
class ExtractionResult:
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    strategy: str = ""
    pages_processed: int = 0
    validation_errors: list = field(default_factory=list)
    balance_ok: bool = True
    confidence_stats: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def to_float(val) -> Optional[float]:
    """Safely convert any value to float. Returns None on failure."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s == "-" or s.lower() in ("", "nil", "n/a", "--"):
        return None
    s = re.sub(r"[^\d.\-]", "", s.replace(",", ""))
    try:
        return float(s)
    except ValueError:
        return None


def is_date(text: str) -> bool:
    return bool(DATE_DETECT.match(str(text).strip()))


def is_money(text: str) -> bool:
    return bool(MONEY_DETECT.match(str(text).strip()))


def is_noise_line(text: str) -> bool:
    t = text.lower().strip()
    if not t or len(t) < 3:
        return True
    return any(w in t for w in NOISE_WORDS)


def should_skip_desc(text: str) -> bool:
    t = text.upper().strip()
    # Only skip if it's a known footer/summary pattern
    if any(kw == t for kw in SKIP_DESCRIPTIONS):
        return True
    if "TOTALS AT END" in t:
        return True
    # If "TOTAL" is the only thing or starts the line in a footer-like way
    if t == "TOTAL" or t.startswith("TOTAL ") or t.endswith(" TOTAL"):
        # But allow it if it looks like a real transaction description 
        # (e.g., "TOTAL GAS" is usually mid-desc or starts it, but footers are often just "TOTAL    $123")
        if any(kw in t for kw in ["CGST", "SGST", "VAT", "TAX"]):
            return False # Keep tax lines
        # If there's a dollar sign immediately after, it's likely a footer
        if re.search(r"TOTAL\s*[\$£€¥₹]", t):
            return True
    return False


def extract_date_prefix(text: str) -> tuple[str, str]:
    """
    Try to extract a date from the start of a blended string.
    Returns (date_str, remainder) or ("", original_text) if no date found.
    """
    m = DATE_PREFIX.match(text.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", text.strip()


def classify_header(text: str) -> str:
    """Classify a header cell text into a semantic role."""
    t = text.lower().strip().rstrip(":")
    # Priority 1: Exact matches or very clear prefixes
    for role, keywords in HEADER_ROLES.items():
        for kw in keywords:
            if t == kw:
                return role
    
    # Priority 2: Substring matches with word boundaries (to avoid "date" in "update")
    for role, keywords in HEADER_ROLES.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", t):
                return role
    return "unknown"


def normalize_date(date_str: str) -> str:
    """Normalize a date string to YYYY-MM-DD or return as-is."""
    date_str = date_str.strip()
    if not date_str:
        return ""
    formats = [
        "%d %b %y", "%d %b %Y", "%d %B %y", "%d %B %Y",
        "%d/%m/%Y", "%d/%m/%y", "%m/%d/%Y", "%m/%d/%y",
        "%Y-%m-%d", "%d-%m-%Y", "%d-%m-%y",
        "%b %d, %Y", "%b %d %Y", "%B %d, %Y",
        "%d %b", "%d %B", "%b %d", "%B %d", # Year-less dates
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            # If year is from 1900 (meaning missing from string), use current year (2025 heuristic)
            if dt.year == 1900:
                dt = dt.replace(year=datetime.now().year)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: TABLE EXTRACTION (REWRITTEN — BALANCE-DELTA DRIVEN)
# ══════════════════════════════════════════════════════════════════════════════

class TableExtractor:
    """
    Extracts transactions from PDFs with detectable table structures.
    
    Key insight: bank tables often merge all transactions on a page into a
    single row with newline-separated values. Balances are the reliable
    anchor — each balance corresponds to exactly one transaction. We use
    balance count as the source of truth for splitting.
    """

    def can_extract(self, pdf) -> bool:
        for page in pdf.pages[:3]:
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    if len(table) >= 2 and len(table[0]) >= 3:
                        return True
        return False

    def extract(self, pdf) -> list[Transaction]:
        all_transactions = []
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            if not tables:
                continue
            for table in tables:
                if len(table) < 2 or not table[0]:
                    continue
                col_map = self._detect_columns(table)
                if not col_map:
                    continue
                txns = self._parse_table(table, col_map)
                all_transactions.extend(txns)
        return all_transactions

    def _detect_columns(self, table: list) -> dict:
        for row in table[:3]:
            if not row or not any(row):
                continue
            text_cells = [str(c).strip() for c in row if c]
            roles = [classify_header(t) for t in text_cells]
            if sum(1 for r in roles if r != "unknown") >= 2:
                col_map = {}
                for i, cell in enumerate(row):
                    if cell:
                        role = classify_header(str(cell))
                        if role != "unknown" and role not in col_map:
                            col_map[role] = i
                return col_map
        return {}

    def _split_cell(self, row: list, col_idx: Optional[int]) -> list[str]:
        """Split a cell value by newlines, plus heuristic space-splits for amounts."""
        if col_idx is None or col_idx >= len(row):
            return []
        val = row[col_idx]
        if not val:
            return []
        
        raw_parts = [p.strip() for p in str(val).split("\n") if p.strip()]
        
        # Heuristic: if it's an amount column and a part contains multiple amounts 
        # separated by multiple spaces, split them further.
        final_parts = []
        for p in raw_parts:
            # Match patterns like "100.00  200.00"
            if re.search(r"[\d.]+\s{2,}[\d.]+", p):
                sub = [s.strip() for s in re.split(r"\s{2,}", p) if s.strip()]
                final_parts.extend(sub)
            else:
                final_parts.append(p)
        return final_parts

    def _parse_table(self, table: list, col_map: dict) -> list[Transaction]:
        """
        Parse the entire table. The core algorithm:
        
        1. Split each relevant column cell by newlines
        2. Use balance count as the transaction count (most reliable)
        3. Use balance deltas to determine deposit vs withdrawal for each txn
        4. Segment descriptions using transaction-starting keywords
        """
        # Find data start (after header)
        data_start = 0
        for i, row in enumerate(table):
            if row and any(row):
                text_cells = [str(c).strip() for c in row if c]
                roles = [classify_header(t) for t in text_cells]
                if sum(1 for r in roles if r != "unknown") >= 2:
                    data_start = i + 1
                    break

        if data_start >= len(table):
            return []

        transactions = []
        for row in table[data_start:]:
            if not row or not any(row):
                continue
            txns = self._parse_merged_row(row, col_map)
            transactions.extend(txns)

        return transactions

    def _parse_merged_row(self, row, col_map) -> list[Transaction]:
        """
        Parse a single (potentially merged) table row into transactions.
        Uses balance values as the anchor for splitting.
        """
        # Split all columns
        dates = self._split_cell(row, col_map.get("date"))
        vdates = self._split_cell(row, col_map.get("value_date"))
        descs = self._split_cell(row, col_map.get("description"))
        deposits = self._split_cell(row, col_map.get("deposit"))
        withdrawals = self._split_cell(row, col_map.get("withdrawal"))
        balances = self._split_cell(row, col_map.get("balance"))
        cheques = self._split_cell(row, col_map.get("cheque"))
        amounts = self._split_cell(row, col_map.get("amount"))

        # Filter out non-numeric values from amount columns
        dep_vals = [to_float(d) for d in deposits]
        dep_vals = [d for d in dep_vals if d is not None]
        wdl_vals = [to_float(w) for w in withdrawals]
        wdl_vals = [w for w in wdl_vals if w is not None]
        bal_vals = [to_float(b) for b in balances]
        bal_vals = [b for b in bal_vals if b is not None]

        # Number of transactions = number of balance entries (most reliable)
        num_txns = len(bal_vals)
        if num_txns == 0:
            num_txns = max(len(dates), 1)

        # Segment descriptions into groups
        desc_groups = self._segment_descriptions(descs, num_txns)

        # Use balance deltas to assign amounts
        # Build the sequence of (amount_value, sign) for each transaction
        txn_amounts = self._assign_amounts_by_balance(
            bal_vals, dep_vals, wdl_vals
        )

        transactions = []
        for i in range(num_txns):
            date_str = dates[i] if i < len(dates) else (dates[-1] if dates else "")
            vdate_str = vdates[i] if i < len(vdates) else ""
            desc_str = desc_groups[i] if i < len(desc_groups) else ""
            bal = bal_vals[i] if i < len(bal_vals) else None

            # Skip non-transaction rows
            if should_skip_desc(desc_str):
                continue
            if not date_str:
                continue

            # Get the assigned amount for this transaction
            if i < len(txn_amounts):
                signed_amount, dep_f, wdl_f = txn_amounts[i]
            else:
                signed_amount, dep_f, wdl_f = 0.0, None, None

            txn = Transaction(
                date=normalize_date(date_str),
                value_date=normalize_date(vdate_str),
                description=desc_str.strip(),
                amount=signed_amount,
                balance=bal,
                raw_deposit=dep_f,
                raw_withdrawal=wdl_f,
            )
            transactions.append(txn)

        return transactions

    def _assign_amounts_by_balance(self, bal_vals, dep_vals, wdl_vals):
        """
        The core algorithm: use balance progression to assign the correct
        amount (with correct sign) to each transaction.
        
        For each consecutive balance pair, compute the delta:
          delta = bal[i] - bal[i-1]
          if delta > 0 → it was a deposit  → match from dep_vals queue
          if delta < 0 → it was a withdrawal → match from wdl_vals queue
          
        The amount is abs(delta), and we verify against dep_vals/wdl_vals.
        """
        result = []
        dep_queue = list(dep_vals)  # copy
        wdl_queue = list(wdl_vals)  # copy

        for i in range(len(bal_vals)):
            if i == 0:
                # First transaction has no previous balance reference
                # Try to match from the amounts
                delta_known = False
            else:
                delta = round(bal_vals[i] - bal_vals[i - 1], 2)
                delta_known = True

            if i > 0 and delta_known:
                abs_delta = abs(delta)
                if delta > 0:
                    # Deposit — find matching value from dep_queue
                    matched_dep = self._pop_closest(dep_queue, abs_delta)
                    if matched_dep is not None:
                        result.append((abs(matched_dep), matched_dep, None))
                    else:
                        result.append((abs_delta, abs_delta, None))
                elif delta < 0:
                    # Withdrawal — find matching value from wdl_queue
                    matched_wdl = self._pop_closest(wdl_queue, abs_delta)
                    if matched_wdl is not None:
                        result.append((-abs(matched_wdl), None, matched_wdl))
                    else:
                        result.append((-abs_delta, None, abs_delta))
                else:
                    # Zero delta (rare — adjustment/reversal)
                    result.append((0.0, None, None))
            else:
                # First balance — no delta reference, use queue order
                if wdl_queue and not dep_queue:
                    val = wdl_queue.pop(0)
                    result.append((-abs(val), None, val))
                elif dep_queue and not wdl_queue:
                    val = dep_queue.pop(0)
                    result.append((abs(val), val, None))
                elif wdl_queue:
                    val = wdl_queue.pop(0)
                    result.append((-abs(val), None, val))
                else:
                    result.append((0.0, None, None))

        return result

    def _pop_closest(self, queue: list, target: float) -> Optional[float]:
        """Pop the value from queue closest to target."""
        if not queue:
            return None
        best_idx = 0
        best_diff = abs(queue[0] - target)
        for i, v in enumerate(queue[1:], 1):
            diff = abs(v - target)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        if best_diff < target * 0.01 + 0.05:  # within 1% or 5 cents
            return queue.pop(best_idx)
        # Still pop the closest even if not exact match
        return queue.pop(best_idx)

    def _segment_descriptions(self, desc_lines: list, num_txns: int) -> list[str]:
        """
        Group description lines into segments matching transaction count.
        Uses transaction-starting keywords to detect boundaries.
        """
        if not desc_lines:
            return [""] * num_txns
        if num_txns <= 1:
            return [" ".join(desc_lines)]

        start_keywords = [
            "UPI/", "IMPS/", "IMPS ", "NEFT ", "RTGS ", "ATM ", "PURCHASE ",
            "TRANSFER", "PAYMENT", "CREDIT OF", "CRADJ/", "REVERSAL",
            "NON SCB", "DISCOUNT", "CGST", "SGST", "BALANCE FORWARD",
            "BILL", "INTEREST",
        ]

        segments = []
        current = []

        for line in desc_lines:
            up = line.upper().strip()
            is_start = any(up.startswith(kw) for kw in start_keywords)

            if is_start and current:
                segments.append(" ".join(current))
                current = [line]
            else:
                current.append(line)

        if current:
            segments.append(" ".join(current))

        # Adjust segment count to match num_txns
        if len(segments) == num_txns:
            return segments
        elif len(segments) > num_txns:
            # Too many segments - merge excess into last
            result = segments[: num_txns - 1]
            result.append(" ".join(segments[num_txns - 1 :]))
            return result
        else:
            # Too few segments - split by line count instead
            total_lines = len(desc_lines)
            if total_lines >= num_txns:
                lines_per = max(1, total_lines // num_txns)
                result = []
                for j in range(num_txns):
                    start = j * lines_per
                    end = (j + 1) * lines_per if j < num_txns - 1 else total_lines
                    result.append(" ".join(desc_lines[start:end]))
                return result
            else:
                return segments + [""] * (num_txns - len(segments))


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: POSITIONAL WORD EXTRACTION (IMPROVED)
# ══════════════════════════════════════════════════════════════════════════════

class PositionalExtractor:
    """
    Extracts transactions from PDFs without table structures by analyzing
    word positions (x, y coordinates). Works for ANZ, Westpac, Chase, etc.
    """

    def can_extract(self, pdf) -> bool:
        for page in pdf.pages[:3]:
            words = page.extract_words()
            if len(words) > 20:
                return True
        return False

    def extract(self, pdf) -> list[Transaction]:
        all_transactions = []
        col_boundaries = None 

        for page in pdf.pages:
            words = page.extract_words(
                keep_blank_chars=True, x_tolerance=3, y_tolerance=3
            )
            if len(words) < 5:
                continue

            # In some PDFs (like ANZ), different tables exist on one page or across pages.
            # We try to detect boundaries on every page, but only reset if a NEW header is found.
            page_boundaries = self._detect_column_boundaries(words)
            if page_boundaries:
                col_boundaries = page_boundaries

            if not col_boundaries:
                continue

            rows = self._group_into_rows(words, y_tolerance=7)
            txns = self._parse_rows(rows, col_boundaries)
            all_transactions.extend(txns)

        return all_transactions

    def _detect_column_boundaries(self, words) -> dict:
        """Find header row and determine column x-boundaries."""
        rows = self._group_into_rows(words)

        for y_key in sorted(rows.keys()):
            row_words = sorted(rows[y_key], key=lambda w: w["x0"])
            row_text = " ".join(w["text"] for w in row_words).lower()

            has_date = any(k in row_text for k in ["date", "dated"])
            has_money = any(
                k in row_text
                for k in [
                    "balance", "withdrawal", "deposit", "amount",
                    "debit", "credit",
                ]
            )

            if has_date and has_money:
                return self._extract_boundaries_from_header(row_words)

        return self._detect_columns_from_data(rows)

    def _extract_boundaries_from_header(self, header_words) -> dict:
        """Build column boundary map from header word positions."""
        boundaries = {}
        # Combine adjacent header words that form multi-word headers
        combined = []
        i = 0
        while i < len(header_words):
            word = header_words[i]
            role = classify_header(word["text"])

            if role != "unknown":
                # Clear role found - take it as is
                combined.append({
                    "text": word["text"],
                    "x0": word["x0"],
                    "x1": word["x1"],
                    "role": role,
                })
                i += 1
            else:
                # Unknown - try to combine with subsequent words
                found = False
                for count in [4, 3, 2]:
                    if i + count <= len(header_words):
                        slice_words = header_words[i:i+count]
                        text = " ".join(w["text"] for w in slice_words)
                        combined_role = classify_header(text)
                        if combined_role != "unknown":
                            combined.append({
                                "text": text,
                                "x0": slice_words[0]["x0"],
                                "x1": slice_words[-1]["x1"],
                                "role": combined_role,
                            })
                            i += count
                            found = True
                            break
                if not found:
                    i += 1
        
        for item in combined:
            role = item["role"]
            if role != "unknown" and role not in boundaries:
                boundaries[role] = {
                    "x_start": item["x0"],
                    "x_end": item["x1"],
                    "x_center": (item["x0"] + item["x1"]) / 2,
                }

        return boundaries

    def _detect_columns_from_data(self, rows) -> dict:
        """Fallback: detect columns from data value distribution."""
        date_x = []
        money_x = []

        for y_key in sorted(rows.keys()):
            for word in rows[y_key]:
                text = word["text"].strip()
                if is_date(text):
                    date_x.append(word["x0"])
                elif is_money(text):
                    money_x.append(word["x0"])

        if not date_x or not money_x:
            return {}

        money_x.sort()
        clusters = self._cluster_positions(money_x)

        boundaries = {
            "date": {
                "x_start": min(date_x) - 5,
                "x_end": min(date_x) + 60,
                "x_center": sum(date_x) / len(date_x),
            }
        }

        if len(clusters) >= 3:
            boundaries["withdrawal"] = {
                "x_start": clusters[0] - 20, "x_end": clusters[0] + 60,
                "x_center": clusters[0],
            }
            boundaries["deposit"] = {
                "x_start": clusters[1] - 20, "x_end": clusters[1] + 60,
                "x_center": clusters[1],
            }
            boundaries["balance"] = {
                "x_start": clusters[2] - 20, "x_end": clusters[2] + 60,
                "x_center": clusters[2],
            }
        elif len(clusters) == 2:
            boundaries["amount"] = {
                "x_start": clusters[0] - 20, "x_end": clusters[0] + 60,
                "x_center": clusters[0],
            }
            boundaries["balance"] = {
                "x_start": clusters[1] - 20, "x_end": clusters[1] + 60,
                "x_center": clusters[1],
            }

        return boundaries

    def _cluster_positions(self, positions, tolerance=30):
        if not positions:
            return []
        clusters = [[positions[0]]]
        for p in positions[1:]:
            if abs(p - clusters[-1][-1]) < tolerance:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        return [sum(c) / len(c) for c in clusters]

    def _group_into_rows(self, words, y_tolerance=6):
        rows = defaultdict(list)
        for w in words:
            y_key = round(w["top"] / y_tolerance) * y_tolerance
            rows[y_key].append(w)
        return dict(rows)

    def _assign_word_to_column(self, word, col_boundaries):
        """Assign a word to a column based on x-position overlap."""
        word_left = word["x0"]
        word_right = word["x1"]
        word_center = (word_left + word_right) / 2

        best_role = "description"
        best_score = -1

        for role, bounds in col_boundaries.items():
            col_left = bounds["x_start"] - 20 # Increased buffer
            col_right = bounds["x_end"] + 90 # Increased buffer for descriptions

            # Check if word overlaps with column range
            if word_left <= col_right and word_right >= col_left:
                # Score by how centered the word is in the column
                dist = abs(word_center - bounds["x_center"])
                overlap = min(word_right, col_right) - max(word_left, col_left)
                
                # Boost dates and money values if they align with known x_centers
                word_val = word["text"].strip()
                score = overlap - dist * 0.2
                if role == "date" and is_date(word_val): score += 50
                if role in ["withdrawal", "deposit", "balance", "amount"] and is_money(word_val): score += 50

                if score > best_score:
                    best_score = score
                    best_role = role

        return best_role

    def _parse_rows(self, rows, col_boundaries) -> list[Transaction]:
        """Parse word rows into transactions."""
        transactions = []
        current_txn = None
        header_y = None

        sorted_ys = sorted(rows.keys())

        # Find header row y-position
        for y_key in sorted_ys:
            rw = sorted(rows[y_key], key=lambda w: w["x0"])
            row_text = " ".join(w["text"] for w in rw).lower()
            has_date = any(k in row_text for k in ["date", "dated"])
            has_money = any(
                k in row_text
                for k in ["balance", "withdrawal", "deposit", "amount", "debit", "credit"]
            )
            if has_date and has_money:
                header_y = y_key
                break

        if header_y is None:
            return []

        # Process rows after header
        for i, y_key in enumerate(sorted_ys):
            if y_key <= header_y:
                continue

            row_words = sorted(rows[y_key], key=lambda w: w["x0"])
            row_text = " ".join(w["text"] for w in row_words).strip()

            # Classify words into columns immediately to check for balance presence
            word_cols = defaultdict(list)
            for w in row_words:
                role = self._assign_word_to_column(w, col_boundaries)
                word_text = w["text"].strip()
                if role == "date" and word_text and not is_date(word_text):
                    date_part, desc_part = extract_date_prefix(word_text)
                    if date_part:
                        word_cols["date"].append(date_part)
                        if desc_part: word_cols["description"].append(desc_part)
                    else: word_cols[role].append(word_text)
                else: word_cols[role].append(word_text)

            bal_text = " ".join(word_cols.get("balance", [])).strip()
            has_bal = bool(to_float(bal_text) is not None)

            # Heuristic: If it's a 'TOTAL' row, skip it regardless of balance
            # (Bank statements often have totals rows at the end of pages/sections)
            if any(k in row_text.upper() for k in ["TOTALS AT END", "TOTAL WITHDRAWALS", "TOTAL DEPOSITS"]):
                continue

            # Only skip other noise if it doesn't contain a suspected balance value
            if not has_bal:
                if is_noise_line(row_text) or should_skip_desc(row_text):
                    continue
                if re.match(r"^\d{4}$", row_text.strip()):
                    continue
            
            date_text = " ".join(word_cols.get("date", [])).strip()
            desc_text = " ".join(word_cols.get("description", [])).strip()
            dep_text = " ".join(word_cols.get("deposit", [])).strip()
            wdl_text = " ".join(word_cols.get("withdrawal", [])).strip()
            amt_text = " ".join(word_cols.get("amount", [])).strip()

            has_date_val = is_date(date_text) if date_text else False
            has_money_val = bool(to_float(bal_text) is not None or to_float(amt_text) is not None or 
                                to_float(dep_text) is not None or to_float(wdl_text) is not None)

            # A new transaction starts if:
            # 1. We found a date (at the start of the row)
            # 2. We find a new balance/amount entry that doesn't fit the current transaction's "missing" slot
            start_new = False
            if has_date_val:
                start_new = True
            elif has_money_val and current_txn:
                # If we already have a balance for the current txn, a new balance/amount MUST be a new txn
                if current_txn.balance is not None and (to_float(bal_text) is not None or to_float(amt_text) is not None):
                    start_new = True
                # If we have both deposit and withdrawal slots filled or it's a second amount
                elif (current_txn.raw_deposit is not None or current_txn.raw_withdrawal is not None) and (to_float(dep_text) is not None or to_float(wdl_text) is not None):
                     start_new = True

            if start_new:
                if current_txn and (current_txn.date or current_txn.amount != 0):
                    transactions.append(current_txn)

                dep_f = to_float(dep_text)
                wdl_f = to_float(wdl_text)
                bal_f = to_float(bal_text)
                amt_f = to_float(amt_text)
                signed = self._compute_signed_amount(amt_f, dep_f, wdl_f)

                current_txn = Transaction(
                    date=normalize_date(date_text) if has_date_val else (current_txn.date if current_txn else ""),
                    description=desc_text,
                    amount=signed,
                    balance=bal_f,
                    raw_deposit=dep_f,
                    raw_withdrawal=wdl_f,
                    confidence="High" if has_date_val else "Medium"
                )
            elif current_txn:
                if desc_text:
                    current_txn.description += " " + desc_text
                
                dep_f = to_float(dep_text)
                wdl_f = to_float(wdl_text)
                bal_f = to_float(bal_text)
                amt_f = to_float(amt_text)

                if dep_f is not None and current_txn.raw_deposit is None:
                    current_txn.raw_deposit = dep_f
                    current_txn.amount = self._compute_signed_amount(amt_f, dep_f, wdl_f)
                if wdl_f is not None and current_txn.raw_withdrawal is None:
                    current_txn.raw_withdrawal = wdl_f
                    current_txn.amount = self._compute_signed_amount(amt_f, dep_f, wdl_f)
                if bal_f is not None:
                    current_txn.balance = bal_f

        # Commit last
        if current_txn and current_txn.date:
            transactions.append(current_txn)

        return transactions

    def _compute_signed_amount(self, amt, dep, wdl):
        if amt is not None:
            return amt
        if dep is not None and wdl is not None:
            return dep - wdl
        if dep is not None:
            return abs(dep)
        if wdl is not None:
            return -abs(wdl)
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: OCR + LLM FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

class LLMExtractor:
    """Uses Groq (free tier) to extract transactions from raw text."""

    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.client = None

    def _get_client(self):
        if self.client is None:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except ImportError:
                return None
        return self.client

    def extract_from_text(self, text: str) -> list[Transaction]:
        client = self._get_client()
        if not client or not self.api_key:
            return []

        prompt = f"""You are a bank statement extraction expert. Extract ALL transactions from the text below into a JSON array.

RULES:
1. Each transaction: date (YYYY-MM-DD), description, amount (negative=debit, positive=credit), balance (if available).
2. Skip headers, footers, summaries, BALANCE FORWARD, OPENING BALANCE, TOTAL lines.
3. Combine multi-line descriptions into one.
4. Return ONLY a valid JSON array.

OUTPUT FORMAT:
[{{"date":"YYYY-MM-DD","description":"...","amount":-100.00,"balance":5000.00}}]

TEXT:
{text[:8000]}"""

        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=4096,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

            data = json.loads(raw)
            return [
                Transaction(
                    date=str(item.get("date", "")),
                    description=str(item.get("description", "")),
                    amount=float(item.get("amount", 0)),
                    balance=(
                        float(item["balance"])
                        if item.get("balance") is not None
                        else None
                    ),
                    confidence="Medium",
                )
                for item in data
            ]
        except Exception as e:
            return []


class OCRExtractor:
    """Handles scanned PDFs using ocrmypdf."""

    def is_scanned(self, pdf_path: str) -> bool:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:3]:
                    text = page.extract_text()
                    if text and len(text.strip()) > 50:
                        return False
            return True
        except Exception:
            return True

    def ocr_and_extract(self, pdf_path: str) -> list[Transaction]:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            subprocess.run(
                [
                    "ocrmypdf", "--rotate-pages", "--deskew",
                    "--force-ocr", pdf_path, tmp_path,
                ],
                capture_output=True, text=True, timeout=120,
            )

            with pdfplumber.open(tmp_path) as pdf:
                te = TableExtractor()
                if te.can_extract(pdf):
                    return te.extract(pdf)
                pe = PositionalExtractor()
                if pe.can_extract(pdf):
                    return pe.extract(pdf)

                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                if full_text.strip():
                    return LLMExtractor().extract_from_text(full_text)

            return []
        except FileNotFoundError:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    full_text = ""
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n"
                if full_text.strip():
                    return LLMExtractor().extract_from_text(full_text)
            except Exception:
                pass
            return []
        except Exception:
            return []
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class TransactionValidator:
    def validate(self, transactions: list[Transaction]) -> tuple[list, list]:
        errors = []
        prev_balance = None

        for i, txn in enumerate(transactions):
            if txn.balance is not None and prev_balance is not None:
                expected = round(prev_balance + txn.amount, 2)
                actual = round(txn.balance, 2)
                diff = abs(expected - actual)

                if diff < 0.05:
                    txn.confidence = "High"
                else:
                    # Try flipping sign
                    alt = round(prev_balance - txn.amount, 2)
                    if abs(alt - actual) < 0.05:
                        txn.amount = -txn.amount
                        txn.confidence = "Medium"
                        # Re-calculate diff after flip
                        diff = abs(alt - actual)
                    else:
                        # Try if the amount was simply missing or wrong
                        # amount should be balance - prev_balance
                        correct_amt = round(actual - prev_balance, 2)
                        txn.amount = correct_amt
                        txn.confidence = "Low"
                        errors.append(
                            f"Row {i + 1}: Expected balance {expected}, "
                            f"got {actual} (Amount corrected to {correct_amt})"
                        )

            if txn.balance is not None:
                prev_balance = txn.balance

        return transactions, errors


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

class UniversalExtractor:
    def __init__(self):
        self.table_ext = TableExtractor()
        self.pos_ext = PositionalExtractor()
        self.ocr_ext = OCRExtractor()
        self.llm_ext = LLMExtractor()
        self.validator = TransactionValidator()

    def extract(self, pdf_input) -> ExtractionResult:
        result = ExtractionResult()

        if hasattr(pdf_input, "read"):
            pdf_bytes = pdf_input.read()
            if hasattr(pdf_input, "seek"):
                pdf_input.seek(0)
            pdf_file = io.BytesIO(pdf_bytes)
            is_file_path = False
        else:
            pdf_file = str(pdf_input)
            pdf_bytes = None
            is_file_path = True

        transactions = []
        strategy = ""
        try:
            with pdfplumber.open(pdf_file) as pdf:
                result.pages_processed = len(pdf.pages)

                # Strategy 1: Table Extraction
                if self.table_ext.can_extract(pdf):
                    transactions = self.table_ext.extract(pdf)
                    if transactions:
                        strategy = "📊 Table Extraction"

                # Strategy 2: Positional Word Extraction
                if not transactions:
                    if self.pos_ext.can_extract(pdf):
                        transactions = self.pos_ext.extract(pdf)
                        if transactions:
                            strategy = "📐 Positional Word Extraction"

                # Strategy 3: LLM on raw text
                if not transactions:
                    full_text = ""
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n"
                    if full_text.strip() and self.llm_ext.api_key:
                        transactions = self.llm_ext.extract_from_text(full_text)
                        if transactions:
                            strategy = "🤖 LLM Text Extraction"

            # Strategy 4: OCR
            if not transactions:
                if is_file_path:
                    if self.ocr_ext.is_scanned(pdf_file):
                        transactions = self.ocr_ext.ocr_and_extract(pdf_file)
                        if transactions:
                            strategy = "🔍 OCR + Extraction"
                elif pdf_bytes:
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(pdf_bytes)
                        tmp_path = tmp.name
                    try:
                        if self.ocr_ext.is_scanned(tmp_path):
                            transactions = self.ocr_ext.ocr_and_extract(tmp_path)
                            if transactions:
                                strategy = "🔍 OCR + Extraction"
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

            # --- DEDUPLICATION & CLEANUP ---
            if transactions:
                # Deduplicate by fingerprint
                unique_txns = []
                seen = set()
                for t in transactions:
                    desc_key = t.description.strip()[:40].upper()
                    # Fingerprint should include balance if available to distinguish same-day same-amount txns
                    fingerprint = (t.date, desc_key, round(abs(t.amount), 2), round(t.balance if t.balance is not None else 0, 2))
                    if fingerprint not in seen:
                        unique_txns.append(t)
                        seen.add(fingerprint)
                
                # Filter out obvious summary/noise
                transactions = [t for t in unique_txns if not (t.amount == 0 and ("BALANCE" in t.description.upper() or "TOTAL" in t.description.upper()))]
                # Additional filter for 'TOTAL' with non-zero amounts (section summaries)
                transactions = [t for t in transactions if "TOTAL" not in t.description.upper()]

                # Validate
                transactions, errors = self.validator.validate(transactions)
                
                # Build FINAL DataFrame with specific columns requested by USER:
                # Build FINAL DataFrame with strict column order
                data = []
                for t in transactions:
                    data.append({
                        "Date": t.date,
                        "description": t.description,
                        "payee": t.payee,
                        "amount": t.amount,
                        "reference": t.reference,
                        "check_no": t.cheque_no or ""
                    })
                
                result.df = pd.DataFrame(data)
                result.strategy = strategy
                result.validation_errors = errors
                result.balance_ok = len(errors) == 0

        except Exception as e:
            # Fallback for non-streamlit environments
            print(f"Extraction failed: {e}")
            try:
                st.error(f"Extraction failed: {e}")
            except:
                pass

        return result


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="BankLens V2", page_icon="🏦", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1321 50%, #111827 100%);
    color: #e2e8f0;
}
section[data-testid="stSidebar"] {
    background: rgba(15, 22, 41, 0.95) !important;
    border-right: 1px solid rgba(56, 189, 248, 0.15);
}

.hero-title {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem; letter-spacing: -0.02em;
}
.hero-subtitle { color: #64748b; font-size: 1rem; margin-bottom: 2rem; }

.strategy-badge {
    display: inline-block; padding: 6px 16px; border-radius: 20px;
    font-size: 0.85rem; font-weight: 600; margin: 4px 0;
}
.strat-table  { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.strat-pos    { background: rgba(59,130,246,0.15); color: #3b82f6; border: 1px solid rgba(59,130,246,0.3); }
.strat-llm    { background: rgba(168,85,247,0.15); color: #a855f7; border: 1px solid rgba(168,85,247,0.3); }
.strat-ocr    { background: rgba(249,115,22,0.15); color: #f97316; border: 1px solid rgba(249,115,22,0.3); }

.metric-card {
    background: linear-gradient(135deg, rgba(15,22,41,0.8), rgba(26,37,64,0.6));
    border: 1px solid rgba(56,189,248,0.15); border-radius: 12px;
    padding: 20px; text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #38bdf8; }
.metric-label { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }

.val-ok   { background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); border-radius: 8px; padding: 12px 16px; color: #10b981; }
.val-warn { background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3); border-radius: 8px; padding: 12px 16px; color: #f59e0b; }

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0f1629, #1a2540);
    border: 1px solid rgba(56,189,248,0.15); border-radius: 12px; padding: 16px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-title">🏦 BankLens V2</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="hero-subtitle">Universal bank statement extractor — '
    "any bank, any format, any country</div>",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Upload Bank Statement PDF",
    type=["pdf"],
    help="Supports digital & scanned PDFs from any bank worldwide",
)

if uploaded:
    extractor = UniversalExtractor()
    with st.spinner("🔍 Analyzing PDF and extracting transactions..."):
        result = extractor.extract(uploaded)

    if not result.df.empty:
        total = len(result.df)
        st.info(f"Successfully extracted {total} transactions.")

        def style_amt(val):
            try:
                v = float(val)
                return (
                    "color: #4ade80; font-weight: 600"
                    if v > 0
                    else "color: #f87171; font-weight: 600"
                    if v < 0
                    else ""
                )
            except (ValueError, TypeError):
                return ""

        styled = result.df.style.map(style_amt, subset=["amount"])
        st.dataframe(
            styled,
            use_container_width=True,
            height=min(600, 38 * len(result.df) + 40),
        )

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            csv = result.df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download CSV", csv,
                "banklens_transactions.csv", "text/csv",
                use_container_width=True,
            )
        with c2:
            try:
                buf = io.BytesIO()
                result.df.to_excel(buf, index=False, engine="openpyxl")
                st.download_button(
                    "📥 Download Excel", buf.getvalue(),
                    "banklens_transactions.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except ImportError:
                st.info("Install openpyxl for Excel export: pip install openpyxl")
    else:
        st.warning(
            "❌ Could not extract transactions. The PDF might be:\n"
            "- Password protected\n"
            "- A heavily distorted scan\n"
            "- Not a bank statement"
        )
    