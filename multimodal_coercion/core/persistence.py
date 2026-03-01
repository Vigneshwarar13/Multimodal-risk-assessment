from pathlib import Path
import json
import sqlite3
import time


class Persistence:
    def __init__(self, db_path, artifacts_dir):
        self.db_path = Path(db_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, ts INTEGER, decision TEXT, prob REAL, meta TEXT)"
        )
        conn.commit()
        conn.close()

    def save_session(self, session_id, decision, prob, meta):
        ts = int(time.time())
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "REPLACE INTO sessions (id, ts, decision, prob, meta) VALUES (?, ?, ?, ?, ?)",
            (session_id, ts, decision, float(prob), json.dumps(meta)),
        )
        conn.commit()
        conn.close()
        j = {"id": session_id, "ts": ts, "decision": decision, "prob": float(prob), "meta": meta}
        p = self.artifacts_dir / f"{session_id}.json"
        with p.open("w", encoding="utf-8") as f:
            json.dump(j, f, ensure_ascii=False, indent=2)
        return str(p)

