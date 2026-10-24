
import sqlite3
from pathlib import Path

# DB 파일 위치: GlucoWise/data/glucowise.db
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "glucowise.db"


def get_connection():
    """SQLite 데이터베이스 연결"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """식사 기록 테이블 생성"""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                demo_user TEXT NOT NULL DEFAULT 'hi',
                timestamp TEXT NOT NULL,
                food TEXT NOT NULL,
                description TEXT DEFAULT '',
                portion_g REAL,
                carbs_g REAL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)


def save_meal(
    timestamp,
    food,
    description="",
    portion_g=None,
    carbs_g=None,
    demo_user="hi",
):
    """새 식사 기록 저장"""
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO meals (
                demo_user, timestamp, food, description, portion_g, carbs_g
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            demo_user,
            str(timestamp),
            food,
            description,
            portion_g,
            carbs_g,
        ))

        return cursor.lastrowid


def get_meals(demo_user="hi"):
    """사용자의 식사 기록을 최신순으로 조회"""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, demo_user, timestamp, food,
                   description, portion_g, carbs_g, created_at
            FROM meals
            WHERE demo_user = ?
            ORDER BY timestamp DESC, id DESC
        """, (demo_user,)).fetchall()

        return [dict(row) for row in rows]


def delete_meal(meal_id, demo_user="hi"):
    """특정 식사 기록 삭제"""
    with get_connection() as conn:
        cursor = conn.execute("""
            DELETE FROM meals
            WHERE id = ? AND demo_user = ?
        """, (meal_id, demo_user))

        return cursor.rowcount > 0
