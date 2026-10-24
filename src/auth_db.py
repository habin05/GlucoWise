"""Local demo login backed by SQLite (stdlib only). Not for deployment.

Each account maps to one synthetic CGM user ("A", "B" or "C").
Passwords are never stored: only a salted PBKDF2 hash.

Setup (once, from the project root):
    python src/auth_db.py add jungyeon A      # asks for a password
    python src/auth_db.py add habin B
    python src/auth_db.py add hyuna C
    python src/auth_db.py list

In app.py (replaces the A/B/C selectbox):
    from auth_db import login_gate
    user_id = login_gate()          # stops the page until logged in
"""
from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "glucowise_users.db"
VALID_USER_IDS = ("A", "B", "C")
ITERATIONS = 200_000
MIN_PASSWORD_LEN = 6


def _connect(db_path=None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
               username TEXT PRIMARY KEY,
               salt BLOB NOT NULL,
               password_hash BLOB NOT NULL,
               cgm_user_id TEXT NOT NULL CHECK (cgm_user_id IN ('A','B','C'))
           )"""
    )
    return conn


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)


def add_user(username: str, password: str, cgm_user_id: str, db_path=None) -> None:
    """Create or overwrite an account. Raises ValueError on bad input."""
    username = username.strip().lower()
    if not username:
        raise ValueError("username is empty")
    if cgm_user_id not in VALID_USER_IDS:
        raise ValueError("cgm_user_id must be A, B or C")
    if len(password) < MIN_PASSWORD_LEN:
        raise ValueError(f"password must be at least {MIN_PASSWORD_LEN} characters")
    salt = os.urandom(16)
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO users VALUES (?, ?, ?, ?)",
            (username, salt, _hash(password, salt), cgm_user_id),
        )


def verify_user(username: str, password: str, db_path=None) -> str | None:
    """Return the CGM user id ("A"/"B"/"C") if the login is right, else None."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT salt, password_hash, cgm_user_id FROM users WHERE username = ?",
            (username.strip().lower(),),
        ).fetchone()
    if row is None:
        _hash(password, b"\x00" * 16)  # similar timing for unknown users
        return None
    salt, stored, cgm_user_id = row
    return cgm_user_id if hmac.compare_digest(_hash(password, salt), stored) else None


def list_users(db_path=None) -> list[tuple[str, str]]:
    with _connect(db_path) as conn:
        return conn.execute(
            "SELECT username, cgm_user_id FROM users ORDER BY username"
        ).fetchall()


def login_gate(db_path=None) -> str:
    """Streamlit gate: shows a login form until success, then returns "A"/"B"/"C"."""
    import streamlit as st

    if st.session_state.get("auth_user_id"):
        with st.sidebar:
            st.write(f"Signed in as **{st.session_state['auth_username']}**")
            if st.button("Log out"):
                for key in ("auth_user_id", "auth_username", "ai_answer"):
                    st.session_state.pop(key, None)
                st.rerun()
        return st.session_state["auth_user_id"]

    st.title("📈 GlucoWise")
    st.info("Demo mode: all glucose data are simulated. Please log in.")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Log in", type="primary"):
        user_id = verify_user(username, password, db_path)
        if user_id:
            st.session_state["auth_user_id"] = user_id
            st.session_state["auth_username"] = username.strip().lower()
            st.rerun()
        else:
            st.error("Wrong username or password.")
    st.stop()


if __name__ == "__main__":
    import getpass

    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "add" and len(sys.argv) == 4:
        pw = getpass.getpass("Password: ")
        if pw != getpass.getpass("Repeat password: "):
            sys.exit("Passwords do not match.")
        try:
            add_user(sys.argv[2], pw, sys.argv[3].upper())
        except ValueError as e:
            sys.exit(f"Error: {e}")
        print(f"Saved account '{sys.argv[2].lower()}' -> CGM user {sys.argv[3].upper()}")
    elif cmd == "list":
        for name, uid in list_users():
            print(f"{name}\t{uid}")
    else:
        print("Usage: python src/auth_db.py add <username> <A|B|C>  |  list")
