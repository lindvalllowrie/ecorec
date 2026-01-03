# log_db.py 
import pandas as pd
import sqlite3
import os
from datetime import datetime

DB_PATH = "recommendation_logs.db"

def init_db():
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS exposure_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_id TEXT,
                target_asin TEXT,
                strategy TEXT,
                rec_asin TEXT,
                rec_rank INTEGER,
                rec_score REAL,
                clicked INTEGER DEFAULT 0,
                converted INTEGER DEFAULT 0
            )
        ''')
        conn.commit()
        conn.close()
        print("✅ 日志数据库初始化完成")

def log_exposure(user_id, target_asin, strategy, rec_list):
    """记录一次推荐曝光"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().isoformat()
    for rank, rec in enumerate(rec_list, 1):
        c.execute('''
            INSERT INTO exposure_log 
            (timestamp, user_id, target_asin, strategy, rec_asin, rec_rank, rec_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            now,
            user_id or "anonymous",
            target_asin or "none",
            strategy,
            rec["ASIN"],
            rank,
            rec.get("total_score", rec.get("similarity", 0))
        ))
    conn.commit()
    conn.close()

def log_click(user_id, rec_asin):
    """记录一次点击（前端调用）—— ✅ 修复 SQLite 语法错误"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        UPDATE exposure_log 
        SET clicked = 1 
        WHERE id = (
            SELECT id FROM exposure_log 
            WHERE user_id = ? AND rec_asin = ? AND clicked = 0
            ORDER BY timestamp DESC 
            LIMIT 1
        )
    ''', (user_id or "anonymous", rec_asin))
    conn.commit()
    conn.close()

def log_convert(user_id, rec_asin):
    """记录一次转化（如加购/下单）—— ✅ 修复 SQLite 语法错误"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        UPDATE exposure_log 
        SET converted = 1 
        WHERE id = (
            SELECT id FROM exposure_log 
            WHERE user_id = ? AND rec_asin = ? AND converted = 0
            ORDER BY timestamp DESC 
            LIMIT 1
        )
    ''', (user_id or "anonymous", rec_asin))
    conn.commit()
    conn.close()

def get_ab_stats():
    """获取A/B测试统计结果"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('''
        SELECT strategy, 
               COUNT(*) as exposures,
               SUM(clicked) as clicks,
               SUM(converted) as conversions
        FROM exposure_log
        GROUP BY strategy
    ''', conn)
    conn.close()
    if len(df) == 0:
        return []
    df["ctr"] = (df["clicks"] / df["exposures"]).round(4)
    df["cvr"] = (df["conversions"] / (df["clicks"] + 1e-6)).round(4)
    return df.to_dict(orient="records")