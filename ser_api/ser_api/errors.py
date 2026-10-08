"""A single exception type for input problems the API should explain, not crash on.
Per SER_Audit_and_Production_Architecture.md section 4: "too short, silent, unreadable,
unsupported format -- never a raw 500." Each SERInputError carries a machine-readable `code`
(stable for clients to branch on) and a human `message` (safe to show a user)."""

from __future__ import annotations


class SERInputError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)
