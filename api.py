# api.py 
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sqlite3
from datetime import datetime
import pandas as pd
import os
import traceback

from main import (
    load_data_with_cache,
    extract_user_info_from_meta,
    build_user_weighted_network,
    get_strategy_for_user,
    smart_recommendation_scheduler
)

# ==================== 日志数据库（修复 SQLite 语法） ====================
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
            rec.get("total_score", 0)
        ))
    conn.commit()
    conn.close()

def log_click(user_id, rec_asin):
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


init_db()

# ==================== FastAPI 主服务 ====================
app = FastAPI(title="EcoRec API", version="1.0")

GLOBAL_FILE_PATHS = {
    "comm_csv": r"D:\bigHW\data\商品-社团映射(1).csv",
    "comm_json": r"D:\bigHW\data\社团-商品列表(1).json",
    "meta_txt": r"D:\bigHW\data\10K_amazon-meta.txt",
    "topology_csv": r"D:\bigHW\data\node_features_10k_topology.csv"
}

full_df = None
tfidf_matrix = None
user_item_dict = None
user_attr_dict = None
user_sim_dict = None
INITIALIZED = False

def initialize():
    global full_df, tfidf_matrix, user_item_dict, user_attr_dict, user_sim_dict, INITIALIZED
    print(" 正在加载推荐系统数据...")
    try:
        full_df, _, tfidf_matrix = load_data_with_cache(GLOBAL_FILE_PATHS)
        user_item_dict, user_attr_dict = extract_user_info_from_meta(full_df, GLOBAL_FILE_PATHS["meta_txt"])
        user_comment_likes = {uid: attr["total_likes"] for uid, attr in user_attr_dict.items()}
        _, user_sim_dict, _ = build_user_weighted_network(user_item_dict, user_comment_likes)
        INITIALIZED = True
        print("✅ 推荐系统初始化完成！")
    except Exception as e:
        print("❌ 初始化失败：")
        traceback.print_exc()

initialize()

# ==================== 接口模型 ====================
class RecommendRequest(BaseModel):
    target_asin: Optional[str] = None
    user_id: Optional[str] = None
    top_n: int = 10

class ClickLogRequest(BaseModel):
    user_id: Optional[str] = None
    asin: str

# ==================== 健康检查 ====================
@app.get("/health")
def health():
    return {"status": "OK", "initialized": INITIALIZED}

@app.get("/ready")
def ready():
    return {"status": "ready"} if INITIALIZED else ({"status": "initializing"}, 503)

# ==================== 推荐与日志 ====================
@app.post("/recommend")
def recommend(req: RecommendRequest):
    if not INITIALIZED:
        return {"code": 503, "message": "服务初始化中", "data": None}
    try:
        strategy = get_strategy_for_user(req.user_id)
        recs = smart_recommendation_scheduler(
            target_asin=req.target_asin,
            target_user_id=req.user_id,
            full_df=full_df,
            tfidf_matrix=tfidf_matrix,
            user_item_dict=user_item_dict,
            user_attr_dict=user_attr_dict,
            user_sim_dict=user_sim_dict,
            file_paths=GLOBAL_FILE_PATHS,  
            top_n=req.top_n
        )
        log_exposure(req.user_id, req.target_asin, strategy, recs)
        return {"code": 200, "message": "成功", "data": {"strategy": strategy, "recommendations": recs}}
    except Exception as e:
        print("❌ 推荐异常：")
        traceback.print_exc()
        return {"code": 500, "message": f"失败: {str(e)}", "data": None}

@app.post("/log/click")
def log_click_api(req: ClickLogRequest):
    try:
        log_click(req.user_id, req.asin)
        return {"code": 200, "message": "点击已记录"}
    except Exception as e:
        return {"code": 500, "message": f"失败: {str(e)}"}

@app.post("/log/convert")
def log_convert_api(req: ClickLogRequest):
    try:
        log_convert(req.user_id, req.asin)
        return {"code": 200, "message": "转化已记录"}
    except Exception as e:
        return {"code": 500, "message": f"失败: {str(e)}"}

@app.get("/ab/stats")
def ab_stats():
    try:
        return {"code": 200, "data": get_ab_stats()}
    except Exception as e:
        return {"code": 500, "message": f"失败: {str(e)}"}

# ==================== 启动 ====================
if __name__ == "__main__":
    import uvicorn
    print(" 启动服务：http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)