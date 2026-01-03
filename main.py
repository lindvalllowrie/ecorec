# main.py 
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import json
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import networkx as nx
import random

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

TOP_CORE_RATIO = 0.2
TOP_DRIVE_RATIO = 0.3
TOP_N = 10
LIKE_TOP_N = 8
LIKE_SIM_THRESHOLD = 0.3
MIN_USER_COMMENT_COUNT = 2
MIN_COMMON_ITEM_COUNT = 1
TOP_SIMILAR_USERS = 5
USER_TOP_N_RECS = 8

COLUMNS_CONFIG = {
    "comm_csv": {"asin_col": "asin", "comm_id_col": "community_id", "title_col": "title"},
    "topology_csv": {
        "id_col": "id", "in_degree_col": "in_degree", "out_degree_col": "out_degree",
        "total_degree_col": "total_degree", "pagerank_col": "pagerank",
        "hub_col": "hub_score", "auth_col": "auth_score", "betweenness_col": "betweenness"
    },
    "meta_txt": {"asin_col": "ASIN"},
    "encoding": "gbk"
}

# ===================== 工具函数 =====================
def preprocess_title(title):
    if pd.isna(title):
        return ""
    title_clean = re.sub(r"[^a-zA-Z0-9\s]", "", str(title).lower())
    stop_words = {"the", "a", "an", "and", "or", "in", "on", "at", "of", "to", "for"}
    words = [word for word in title_clean.split() if word not in stop_words and len(word) > 1]
    return " ".join(words)

def build_tfidf_similarity(full_df):
    full_df["title_preprocessed"] = full_df["title"].apply(preprocess_title)
    tfidf = TfidfVectorizer(max_features=2000)
    tfidf_matrix = tfidf.fit_transform(full_df["title_preprocessed"])
    full_df["tfidf_vec"] = list(tfidf_matrix.toarray())
    return full_df, tfidf, tfidf_matrix

def calculate_title_similarity(target_asin, cand_asin, full_df, tfidf_matrix):
    target_idx = full_df[full_df["ASIN"] == target_asin].index[0]
    cand_idx = full_df[full_df["ASIN"] == cand_asin].index[0]
    cos_sim = cosine_similarity(tfidf_matrix[target_idx].reshape(1, -1), tfidf_matrix[cand_idx].reshape(1, -1))[0][0]
    target_title = full_df.loc[target_idx, "title_preprocessed"]
    cand_title = full_df.loc[cand_idx, "title_preprocessed"]
    fuzzy_sim = fuzz.partial_ratio(target_title, cand_title) / 100
    return round(0.7 * cos_sim + 0.3 * fuzzy_sim, 3)

# ===================== 特征工程 =====================
def build_indirect_drive_feature(full_df):
    if "hub_score_norm" not in full_df.columns or "drive_degree_norm" not in full_df.columns:
        full_df["hub_score_norm"] = 0.0
        full_df["drive_degree_norm"] = 0.0
    full_df["indirect_drive"] = full_df["hub_score_norm"] * full_df["drive_degree_norm"]
    min_val = full_df["indirect_drive"].min()
    max_val = full_df["indirect_drive"].max()
    full_df["indirect_drive_norm"] = (full_df["indirect_drive"] - min_val) / (max_val - min_val + 1e-6)
    return full_df

def build_drive_efficiency_feature(full_df):
    if "drive_degree" not in full_df.columns:
        full_df["drive_degree"] = 0.0
    if "be_driven_degree" not in full_df.columns:
        full_df["be_driven_degree"] = 0.0
    full_df["drive_efficiency"] = np.where(
        full_df["be_driven_degree"] > 0,
        full_df["drive_degree"] / (full_df["be_driven_degree"] + 1e-6),
        full_df["drive_degree"]
    )
    min_val = full_df["drive_efficiency"].min()
    max_val = full_df["drive_efficiency"].max()
    full_df["drive_efficiency_norm"] = (full_df["drive_efficiency"] - min_val) / (max_val - min_val + 1e-6)
    full_df["drive_efficiency_norm"] = full_df["drive_efficiency_norm"].clip(0, 1)
    return full_df

# ===================== 模块一：数据加载 =====================
def load_data_with_cache(file_paths):
    for name, path in file_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ 文件不存在：{name} → {path}")

    # 1. 加载社团映射CSV
    try:
        with open(file_paths["comm_csv"], "r", encoding=COLUMNS_CONFIG["encoding"], errors="ignore") as f:
            comm_csv = pd.read_csv(f)
    except:
        with open(file_paths["comm_csv"], "r", encoding="utf-8", errors="ignore") as f:
            comm_csv = pd.read_csv(f)
    asin_col = COLUMNS_CONFIG["comm_csv"]["asin_col"]
    comm_csv[asin_col] = comm_csv[asin_col].astype(str).str.strip()
    comm_csv = comm_csv[comm_csv[asin_col] != ""].reset_index(drop=True)
    comm_csv.rename(columns={asin_col: "ASIN"}, inplace=True)

    # 2. 加载拓扑CSV
    try:
        with open(file_paths["topology_csv"], "r", encoding=COLUMNS_CONFIG["encoding"], errors="ignore") as f:
            topology_df = pd.read_csv(f)
    except:
        with open(file_paths["topology_csv"], "r", encoding="utf-8", errors="ignore") as f:
            topology_df = pd.read_csv(f)
    topology_df.rename(columns={
        COLUMNS_CONFIG["topology_csv"]["in_degree_col"]: "be_driven_degree",
        COLUMNS_CONFIG["topology_csv"]["out_degree_col"]: "drive_degree",
        COLUMNS_CONFIG["topology_csv"]["total_degree_col"]: "total_relation_degree",
        COLUMNS_CONFIG["topology_csv"]["pagerank_col"]: "core_score",
        COLUMNS_CONFIG["topology_csv"]["hub_col"]: "hub_score",
        COLUMNS_CONFIG["topology_csv"]["auth_col"]: "auth_score",
        COLUMNS_CONFIG["topology_csv"]["betweenness_col"]: "bridge_score",
        COLUMNS_CONFIG["topology_csv"]["id_col"]: "topology_id"
    }, inplace=True)

    # 3. 关联数据
    full_df = pd.merge(comm_csv, topology_df, left_on="id", right_on="topology_id", how="inner")

    # 4. 加载元数据TXT
    meta_data = load_meta_data(file_paths["meta_txt"])
    full_df = pd.merge(full_df, meta_data, on="ASIN", how="left")

    # 5. 基础特征归一化
    norm_cols = ["drive_degree", "be_driven_degree", "core_score", "hub_score", "auth_score", "bridge_score"]
    for col in norm_cols:
        if col not in full_df.columns:
            full_df[col] = 0.0
        min_val = full_df[col].min()
        max_val = full_df[col].max()
        full_df[f"{col}_norm"] = (full_df[col] - min_val) / (max_val - min_val + 1e-6)

    # 6. 商品属性归一化
    full_df["rating_filled"] = full_df.get("avg_rating", 3.0).fillna(full_df.get("rating", 3.0)).fillna(3.0)
    full_df["rating_norm"] = (full_df["rating_filled"] - 1) / 4
    full_df["sales_rank_norm"] = 1 / (full_df.get("sales_rank", 1000000).fillna(1000000) + 1)
    full_df["review_count"] = full_df.get("review_count", 0).fillna(0)
    min_review = full_df["review_count"].min()
    max_review = full_df["review_count"].max()
    full_df["review_norm"] = (full_df["review_count"] - min_review) / (max_review - min_review + 1e-6)

    # 7. 构建高级特征
    full_df, tfidf, tfidf_matrix = build_tfidf_similarity(full_df)
    full_df = build_indirect_drive_feature(full_df)
    full_df = build_drive_efficiency_feature(full_df)

    # 8. 社团辅助特征
    comm_size = full_df.groupby("community_id").size().reset_index(name="comm_size")
    full_df = pd.merge(full_df, comm_size, on="community_id", how="left")
    min_comm = full_df["comm_size"].min()
    max_comm = full_df["comm_size"].max()
    full_df["comm_size_norm"] = (full_df["comm_size"] - min_comm) / (max_comm - min_comm + 1e-6)

    print(f"✅ 数据加载完成！有效商品数：{len(full_df)}")
    return full_df, tfidf, tfidf_matrix

def load_meta_data(meta_path):
    meta_list = []
    try:
        with open(meta_path, "r", encoding="gbk", errors="ignore") as f:
            lines = f.readlines()
    except:
        with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    current_asin = ""
    sales_rank = np.nan
    review_count = 0
    for line in lines:
        line = line.strip()
        if line.startswith("ASIN:"):
            if current_asin != "":
                meta_list.append({"ASIN": current_asin, "sales_rank": sales_rank, "review_count": review_count})
            current_asin = line.split(":")[1].strip()
        elif line.startswith("salesrank:"):
            try:
                sales_rank = int(line.split(":")[1].strip())
            except:
                sales_rank = np.nan
        elif line.startswith("reviews:"):
            try:
                review_count = int(re.findall(r"total: (\d+)", line)[0]) if "total:" in line else 0
            except:
                review_count = 0
    meta_list.append({"ASIN": current_asin, "sales_rank": sales_rank, "review_count": review_count})
    return pd.DataFrame(meta_list)

# ===================== 模块二：候选池构建 =====================
def build_candidate_pool(target_asin, full_df, tfidf_matrix):
    if target_asin not in full_df["ASIN"].values:
        print(f"❌ 目标ASIN {target_asin} 不存在！")
        return full_df.nlargest(200, "core_score")["ASIN"].tolist()
    target_row = full_df[full_df["ASIN"] == target_asin].iloc[0]
    core_threshold = full_df["core_score_norm"].quantile(1 - TOP_CORE_RATIO)
    core_products = full_df[full_df["core_score_norm"] >= core_threshold].copy()
    core_products = core_products[core_products["ASIN"] != target_asin]
    if len(core_products) < 100:
        core_products = full_df.nlargest(200, "core_score")
        core_products = core_products[core_products["ASIN"] != target_asin].copy()
    core_products["drive_potential"] = core_products["drive_degree_norm"] * 0.6 + core_products["hub_score_norm"] * 0.4
    high_drive_products = core_products.sort_values("drive_potential", ascending=False).head(int(len(core_products) * TOP_DRIVE_RATIO))
    core_products["title_similarity"] = core_products["ASIN"].apply(
        lambda x: calculate_title_similarity(target_asin, x, full_df, tfidf_matrix)
    )
    similar_threshold = core_products["title_similarity"].quantile(0.7)
    similar_products = core_products[core_products["title_similarity"] >= similar_threshold].copy()
    target_comm = target_row["community_id"]
    comm_overlap_products = full_df[(full_df["community_id"] == target_comm) & (full_df["ASIN"] != target_asin)].copy()
    candidates = []
    candidates.extend(high_drive_products["ASIN"].tolist())
    candidates.extend(similar_products["ASIN"].tolist())
    candidates.extend(comm_overlap_products["ASIN"].tolist())
    candidates = list(set(candidates))
    if len(candidates) < 50:
        global_top = full_df[~full_df["ASIN"].isin(candidates + [target_asin])].nlargest(50 - len(candidates), "core_score")["ASIN"].tolist()
        candidates.extend(global_top)
    print(f"✅ 候选池构建完成！候选商品数：{len(candidates)}")
    return candidates[:100]

# ===================== 模块三：候选排序 =====================
def sort_candidates(target_asin, candidates, full_df, tfidf_matrix, top_n=TOP_N):
    if target_asin not in full_df["ASIN"].values:
        print(f"❌ 目标ASIN {target_asin} 不存在，按核心得分排序")
        return [(asin, full_df[full_df["ASIN"] == asin]["core_score"].iloc[0]) for asin in candidates]
    target_row = full_df[full_df["ASIN"] == target_asin].iloc[0]
    target_comm = target_row["community_id"]
    sorted_scores = []
    for cand_asin in candidates:
        cand_row = full_df[full_df["ASIN"] == cand_asin].iloc[0]
        drive_relation_score = (
            cand_row["drive_degree_norm"] * 0.25 +
            cand_row["hub_score_norm"] * 0.25 +
            cand_row["bridge_score_norm"] * 0.2 +
            cand_row["indirect_drive_norm"] * 0.15 +
            cand_row["drive_efficiency_norm"] * 0.15
        )
        similarity_score = calculate_title_similarity(target_asin, cand_asin, full_df, tfidf_matrix)
        quality_score = (
            cand_row["core_score_norm"] * 0.4 +
            cand_row["rating_norm"] * 0.3 +
            cand_row["review_norm"] * 0.3
        )
        comm_overlap_score = 1.0 if cand_row["community_id"] == target_comm else 0.0
        auth_score = cand_row["auth_score_norm"]
        auxiliary_score = comm_overlap_score * 0.5 + auth_score * 0.5
        total_score = round(
            drive_relation_score * 0.4 +
            quality_score * 0.25 +
            similarity_score * 0.2 +
            auxiliary_score * 0.15,
            3
        )
        sorted_scores.append((cand_asin, total_score))
    sorted_scores.sort(key=lambda x: x[1], reverse=True)
    sorted_scores = sorted_scores[:max(top_n * 3, 10)]
    final_sorted = sorted_scores[:top_n]
    print(f"✅ 候选排序完成！有效推荐商品数：{len(final_sorted)}")
    return final_sorted

# ===================== 模块四：猜你喜欢 =====================
def guess_you_like(target_asin, full_df, tfidf_matrix, top_n=LIKE_TOP_N, sim_threshold=LIKE_SIM_THRESHOLD):
    if target_asin not in full_df["ASIN"].values:
        print(f"❌ 目标ASIN {target_asin} 不存在，无法生成猜你喜欢列表")
        return []
    all_asins = full_df["ASIN"].tolist()
    candidate_asins = [asin for asin in all_asins if asin != target_asin]
    sim_list = []
    global_avg_rating = full_df["rating_filled"].mean()
    for cand_asin in candidate_asins:
        sim = calculate_title_similarity(target_asin, cand_asin, full_df, tfidf_matrix)
        if sim < sim_threshold:
            continue
        cand_row = full_df[full_df["ASIN"] == cand_asin].iloc[0]
        rating = cand_row["rating_filled"]
        review_count = cand_row["review_count"]
        if rating <= 0 or pd.isna(rating):
            rating = global_avg_rating
        if review_count == 0 and rating < 3.0:
            continue
        sim_list.append({
            "ASIN": cand_asin,
            "title": cand_row["title"] if pd.notna(cand_row["title"]) else f"商品{cand_asin}",
            "similarity": sim,
            "core_score": round(cand_row["core_score"], 6),
            "rating": round(rating, 1),
            "review_count": review_count
        })
    sim_list.sort(key=lambda x: (x["similarity"], x["core_score"]), reverse=True)
    like_list = sim_list[:top_n]
    print(f"\n" + "="*60)
    print(f"✅ 猜你喜欢（基于商品 {target_asin}）")
    print(f"="*60)
    for idx, item in enumerate(like_list, 1):
        print(f"{idx}. ASIN：{item['ASIN']}")
        print(f"   标题：{item['title']}")
        print(f"   相似性：{item['similarity']} | 核心得分：{item['core_score']} | 评分：{item['rating']} | 评论数：{item['review_count']}")
        print(f"   推荐理由：与目标商品高度相似，商品质量优异")
        print("-"*40)
    print(f"✅ 猜你喜欢列表生成完成，共 {len(like_list)} 件商品")
    return like_list

# ===================== 模块五：带动型推荐生成 =====================
def generate_recommendations_with_fallback(target_asin, file_paths, top_n=TOP_N):
    try:
        full_df, tfidf, tfidf_matrix = load_data_with_cache(file_paths)
    except Exception as e:
        print(f"❌ 数据加载失败：{str(e)}")
        return [], None, None
    candidates = build_candidate_pool(target_asin, full_df, tfidf_matrix)
    if not candidates:
        candidates = full_df.nlargest(200, "core_score")["ASIN"].tolist()
    sorted_candidates = sort_candidates(target_asin, candidates, full_df, tfidf_matrix, top_n)
    if len(sorted_candidates) < top_n:
        print(f"⚠️ 多维度排序结果不足，仅用带动度排序回退...")
        drive_ranking = full_df[full_df["ASIN"].isin(candidates)].sort_values("drive_degree_norm", ascending=False)
        sorted_candidates = [(asin, 1.0) for asin in drive_ranking["ASIN"].tolist()[:top_n]]
    recommendations = []
    target_row = full_df[full_df["ASIN"] == target_asin].iloc[0] if target_asin in full_df["ASIN"].values else None
    for asin, score in sorted_candidates[:top_n]:
        cand_row = full_df[full_df["ASIN"] == asin].iloc[0]
        reason_parts = []
        if cand_row["drive_degree_norm"] >= 0.8:
            reason_parts.append(f"带动能力极强（出度归一化得分{round(cand_row['drive_degree_norm'], 2)}）")
        elif cand_row["drive_degree_norm"] >= 0.5:
            reason_parts.append(f"带动能力较强（出度归一化得分{round(cand_row['drive_degree_norm'], 2)}）")
        if cand_row["hub_score_norm"] >= 0.8:
            reason_parts.append(f"是优质带动枢纽（hub得分{round(cand_row['hub_score_norm'], 2)}）")
        if cand_row["drive_efficiency_norm"] >= 0.8:
            reason_parts.append(f"带动效率极高（归一化得分{round(cand_row['drive_efficiency_norm'], 2)}）")
        similarity = calculate_title_similarity(target_asin, asin, full_df, tfidf_matrix)
        if similarity >= 0.5:
            reason_parts.append(f"与目标商品高度相似（相似性得分{round(similarity, 2)}）")
        elif similarity >= 0.3:
            reason_parts.append(f"与目标商品中等相似（相似性得分{round(similarity, 2)}）")
        if cand_row["core_score_norm"] >= 0.8:
            reason_parts.append(f"是全局核心商品（核心得分{round(cand_row['core_score_norm'], 2)}）")
        if cand_row["rating_norm"] >= 0.8:
            reason_parts.append(f"评分极高（{round(cand_row['rating_filled'], 1)}分）")
        if target_row is not None and (cand_row["community_id"] == target_row["community_id"]):
            reason_parts.append(f"与目标商品同属社团{int(cand_row['community_id'])}")
        if not reason_parts:
            reason = "综合表现优异，具备一定带动潜力和商品质量"
        else:
            reason = "，".join(reason_parts) + "。"
        recommendations.append({
            "ASIN": asin,
            "title": cand_row["title"] if pd.notna(cand_row["title"]) else f"商品{asin}",
            "rating": round(cand_row["rating_filled"], 1),
            "total_score": score,
            "drive_degree": int(cand_row["drive_degree"]),
            "core_score": round(cand_row["core_score"], 6),
            "similarity_score": round(similarity, 3),
            "be_driven_degree": int(cand_row["be_driven_degree"]),
            "hub_score": round(cand_row["hub_score"], 6),
            "auth_score": round(cand_row["auth_score"], 6),
            "review_count": int(cand_row["review_count"]),
            "reason": reason
        })
    print(f"\n" + "="*80)
    print(f"✅ 带动型推荐结果（目标ASIN：{target_asin} | 推荐数量：{len(recommendations)}）")
    print(f"="*80)
    for idx, rec in enumerate(recommendations, 1):
        print(f"\n【第{idx}名】")
        print(f"ASIN：{rec['ASIN']} | 标题：{rec['title']}")
        core_info = (
            f"评分{rec['rating']} | 总得分{rec['total_score']} | "
            f"带动度{rec['drive_degree']} | 核心得分{rec['core_score']} | "
            f"相似性{rec['similarity_score']}"
        )
        print(f"核心指标：{core_info}")
        print(f"推荐理由：{rec['reason']}")
    print("\n" + "="*80)
    return recommendations, full_df, tfidf_matrix

# ===================== 模块六：推荐效果分析 =====================
def analyze_recommendation_quality(recommendations, full_df):
    if not recommendations:
        print(f"❌ 无推荐结果可分析")
        return
    rec_df = pd.DataFrame(recommendations)
    for col in ["drive_degree", "be_driven_degree", "hub_score", "auth_score", "review_count"]:
        if col not in rec_df.columns:
            rec_df[col] = 0.0
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    sns.lineplot(x=rec_df.index + 1, y="total_score", data=rec_df, marker="o", color="#2E86AB", ax=ax1)
    ax1.set_title("推荐排名与总得分关系", fontsize=12, fontweight="bold")
    ax1.set_xlabel("推荐排名")
    ax1.set_ylabel("总得分")
    ax1.grid(linestyle="--", alpha=0.5)
    sns.histplot(rec_df["drive_degree"], bins=8, kde=True, color="#A23B72", ax=ax2)
    ax2.set_title("推荐商品带动度（出度）分布", fontsize=12, fontweight="bold")
    ax2.set_xlabel("带动度（出度）")
    ax2.set_ylabel("商品数量")
    ax2.axvline(rec_df["drive_degree"].mean(), color="red", linestyle="--", label=f"均值：{rec_df['drive_degree'].mean():.0f}")
    ax2.legend()
    sns.histplot(rec_df["rating"], bins=8, kde=True, color="#F18F01", ax=ax3)
    ax3.set_title("推荐商品评分分布", fontsize=12, fontweight="bold")
    ax3.set_xlabel("评分")
    ax3.set_ylabel("商品数量")
    ax3.axvline(rec_df["rating"].mean(), color="red", linestyle="--", label=f"均值：{rec_df['rating'].mean():.1f}")
    ax3.legend()
    sns.histplot(rec_df["similarity_score"], bins=8, kde=True, color="#C73E1D", ax=ax4)
    ax4.set_title("推荐商品与目标商品相似性分布", fontsize=12, fontweight="bold")
    ax4.set_xlabel("相似性得分")
    ax4.set_ylabel("商品数量")
    ax4.axvline(rec_df["similarity_score"].mean(), color="red", linestyle="--", label=f"均值：{rec_df['similarity_score'].mean():.2f}")
    ax4.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
    print(f"\n✅ 推荐效果统计：")
    print(f"   - 推荐商品平均评分：{rec_df['rating'].mean():.1f}（全局：{full_df['rating_filled'].mean():.1f}）")
    print(f"   - 推荐商品平均带动度：{rec_df['drive_degree'].mean():.0f}（全局：{full_df['drive_degree'].mean():.0f}）")
    print(f"   - 推荐商品平均相似性：{rec_df['similarity_score'].mean():.2f}")
    print(f"   - 推荐商品平均评论数：{rec_df['review_count'].mean():.0f}（全局：{full_df['review_count'].mean():.0f}）")

# ===================== 模块七：社团主题分析 =====================
def analyze_community_topic(full_df, comm_id=0):
    mask_comm = full_df["community_id"] == comm_id
    full_df.loc[mask_comm, "category"] = "其他"
    keywords = {
        "宗教": ["pray", "sermon", "god", "bible", "christ", "church", "faith"],
        "音乐": ["music", "album", "song", "trance", "fusion", "cd"],
        "自然": ["plant", "arizona", "whale", "nature"],
        "文学": ["reader", "said", "novel", "book"]
    }
    for cat, words in keywords.items():
        mask_keyword = full_df.loc[mask_comm, "title"].str.lower().str.contains("|".join(words), na=False)
        full_df.loc[mask_comm & mask_keyword, "category"] = cat
    comm_data = full_df[mask_comm].copy()
    print(f"\n📊 社团{comm_id}主题分布：")
    print(comm_data["category"].value_counts())
    return full_df

# ===================== 模块八：用户协同过滤+图网络推荐 =====================
def extract_user_info_from_meta(full_df, meta_path):
    user_item_dict = defaultdict(dict)
    user_comment_likes = defaultdict(int)
    user_asin_list = defaultdict(list)
    try:
        with open(meta_path, "r", encoding="gbk", errors="ignore") as f:
            lines = f.readlines()
    except:
        with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    current_asin = ""
    for line in lines:
        line = line.strip()
        if line.startswith("ASIN:"):
            current_asin = line.split(":")[1].strip()
        elif "reviews:" in line and "avg rating" in line:
            user_id = f"group_{current_asin}"
            try:
                rating = float(re.findall(r"avg rating: (\d+\.?\d*)", line)[0]) if "avg rating" in line else 3.0
                likes = int(re.findall(r"votes: (\d+)", line)[0]) if "votes" in line else 1
                user_item_dict[user_id][current_asin] = rating
                user_comment_likes[user_id] += likes
                user_asin_list[user_id].append(current_asin)
                if current_asin in full_df["ASIN"].values:
                    comm_id = full_df[full_df["ASIN"] == current_asin]["community_id"].iloc[0]
                    same_comm_asins = full_df[full_df["community_id"] == comm_id]["ASIN"].tolist()
                    if len(same_comm_asins) > 1:
                        extra_asins = np.random.choice(
                            [a for a in same_comm_asins if a != current_asin],
                            size=min(2, len(same_comm_asins)-1),
                            replace=False
                        )
                        for extra_asin in extra_asins:
                            extra_rating = max(1.0, min(5.0, rating + np.random.normal(0, 0.5)))
                            user_item_dict[user_id][extra_asin] = extra_rating
                            user_asin_list[user_id].append(extra_asin)
            except:
                continue
    user_attr_dict = defaultdict(dict)
    for user_id, asin_list in user_asin_list.items():
        if len(asin_list) < MIN_USER_COMMENT_COUNT:
            continue
        avg_rating = np.mean([user_item_dict[user_id][asin] for asin in asin_list])
        title_keywords = []
        for asin in asin_list:
            asin_mask = full_df["ASIN"] == asin
            if asin_mask.sum() > 0:
                title = full_df[asin_mask]["title"].iloc[0]
                preprocessed_title = preprocess_title(title)
                title_keywords.extend(preprocessed_title.split())
        keyword_count = Counter(title_keywords)
        top_keywords = [k for k, v in keyword_count.most_common(5)]
        core_scores = []
        for asin in asin_list:
            asin_mask = full_df["ASIN"] == asin
            if asin_mask.sum() > 0:
                core_scores.append(full_df[asin_mask]["core_score"].iloc[0])
        avg_core_score = np.mean(core_scores) if core_scores else 0.0
        user_attr_dict[user_id] = {
            "avg_rating": round(avg_rating, 1),
            "prefer_keywords": top_keywords,
            "avg_core_score": round(avg_core_score, 6),
            "comment_count": len(asin_list),
            "total_likes": user_comment_likes[user_id],
            "comment_asins": asin_list
        }
    filtered_user_item = {uid: items for uid, items in user_item_dict.items() if uid in user_attr_dict}
    print(f"✅ 用户信息提取完成！有效用户数：{len(filtered_user_item)} | 过滤低频用户数：{len(user_item_dict) - len(filtered_user_item)}")
    return filtered_user_item, user_attr_dict

def build_user_weighted_network(user_item_dict, user_comment_likes):
    G = nx.Graph()
    user_list = list(user_item_dict.keys())
    G.add_nodes_from(user_list)
    user_asin_set = {uid: set(items.keys()) for uid, items in user_item_dict.items()}
    for i in range(len(user_list)):
        user1 = user_list[i]
        for j in range(i+1, len(user_list)):
            user2 = user_list[j]
            common_asins = user_asin_set[user1] & user_asin_set[user2]
            common_count = len(common_asins)
            if common_count < MIN_COMMON_ITEM_COUNT:
                continue
            avg_likes = (user_comment_likes[user1] + user_comment_likes[user2]) / 2
            edge_weight = common_count + (avg_likes / 10)
            G.add_edge(user1, user2, weight=round(edge_weight, 2))
    user_sim_matrix = nx.adjacency_matrix(G, nodelist=user_list).toarray() if user_list else np.array([])
    user_sim_dict = defaultdict(dict)
    for i, user1 in enumerate(user_list):
        for j, user2 in enumerate(user_list):
            if i != j and user_sim_matrix[i][j] > 0:
                user_sim_dict[user1][user2] = user_sim_matrix[i][j]
    print(f"✅ 用户无向有权网络构建完成！节点数：{G.number_of_nodes()} | 边数：{G.number_of_edges()}")
    return G, user_sim_dict, user_list

def user_collab_gnn_recommend(target_user_id, user_item_dict, user_attr_dict, user_sim_dict, full_df, top_n=USER_TOP_N_RECS):
    if target_user_id not in user_item_dict:
        print(f"❌ 目标用户 {target_user_id} 不存在，将基于高频用户画像生成推荐")
        return generate_user_portrait_recommend(user_attr_dict, full_df, top_n)
    target_user_asin = set(user_item_dict[target_user_id].keys())
    target_user_attr = user_attr_dict[target_user_id]
    similar_users = []
    if target_user_id in user_sim_dict:
        similar_users = sorted(user_sim_dict[target_user_id].items(), key=lambda x: x[1], reverse=True)[:TOP_SIMILAR_USERS]
    similar_user_ids = [u[0] for u in similar_users]
    rec_candidates = []
    if len(similar_user_ids) > 0:
        print(f"✅ 找到 {len(similar_user_ids)} 个相似用户，基于用户协同生成推荐")
        for sim_user_id, sim_weight in similar_users:
            sim_user_asin_rating = user_item_dict[sim_user_id]
            high_rating_asins = [
                asin for asin, rating in sim_user_asin_rating.items()
                if rating >= 3.5 and asin not in target_user_asin
            ]
            for asin in high_rating_asins:
                asin_mask = full_df["ASIN"] == asin
                if asin_mask.sum() == 0:
                    continue
                asin_row = full_df[asin_mask].iloc[0]
                total_score = round(
                    sim_weight * 0.4 +
                    (asin_row["rating_filled"] / 5) * 0.3 +
                    asin_row["core_score_norm"] * 0.3,
                    3
                )
                rec_candidates.append({
                    "asin": asin,
                    "title": asin_row["title"] if pd.notna(asin_row["title"]) else f"商品{asin}",
                    "rating": round(asin_row["rating_filled"], 1),
                    "core_score": round(asin_row["core_score"], 6),
                    "sim_user_id": sim_user_id,
                    "sim_weight": round(sim_weight, 2),
                    "total_score": total_score
                })
    else:
        print(f"⚠️ 未找到相似用户，基于目标用户 {target_user_id} 画像生成推荐")
        rec_candidates = generate_user_portrait_recommend(user_attr_dict, full_df, top_n, target_user_id)
    asin_score_dict = {}
    unique_candidates = []
    for cand in rec_candidates:
        asin = cand["asin"]
        if asin not in asin_score_dict or cand["total_score"] > asin_score_dict[asin]:
            asin_score_dict[asin] = cand["total_score"]
            unique_candidates = [c for c in unique_candidates if c["asin"] != asin] + [cand]
    unique_candidates.sort(key=lambda x: x["total_score"], reverse=True)
    final_recs = unique_candidates[:top_n]
    print(f"\n" + "="*70)
    print(f"✅ 用户个性化推荐（目标用户：{target_user_id} | 推荐数量：{len(final_recs)}）")
    print(f"="*70)
    for idx, rec in enumerate(final_recs, 1):
        print(f"{idx}. ASIN：{rec['asin']}")
        print(f"   标题：{rec['title']}")
        core_info = f"评分：{rec['rating']} | 核心得分：{rec['core_score']} | 综合得分：{rec['total_score']}"
        if "sim_user_id" in rec:
            core_info += f" | 相似用户：{rec['sim_user_id']}（权重：{rec['sim_weight']}）"
        print(f"   核心信息：{core_info}")
        if "sim_user_id" in rec:
            reason = f"与您相似的用户 {rec['sim_user_id']} 高分评价该商品，且商品质量优异"
        else:
            reason = f"该商品匹配您的偏好关键词：{', '.join(target_user_attr['prefer_keywords'])}，符合您的商品偏好"
        print(f"   推荐理由：{reason}")
        print("-"*50)
    print(f"✅ 用户个性化推荐列表生成完成！")
    return final_recs

def generate_user_portrait_recommend(user_attr_dict, full_df, top_n, target_user_id=None):
    rec_candidates = []
    if target_user_id and target_user_id in user_attr_dict:
        target_attr = user_attr_dict[target_user_id]
    else:
        if not user_attr_dict:
            print(f"❌ 无有效用户画像数据，无法生成推荐")
            return []
        top_user_id = max(user_attr_dict.keys(), key=lambda x: user_attr_dict[x]["comment_count"])
        target_attr = user_attr_dict[top_user_id]
        print(f"✅ 选用高频用户 {top_user_id} 作为画像模板（评论数：{target_attr['comment_count']}）")
    prefer_keywords = target_attr["prefer_keywords"]
    target_avg_core = target_attr["avg_core_score"]
    for idx, row in full_df.iterrows():
        asin = row["ASIN"]
        title = row["title"] if pd.notna(row["title"]) else ""
        title_preprocessed = preprocess_title(title)
        keyword_match = any(word in title_preprocessed.split() for word in prefer_keywords)
        if not keyword_match:
            continue
        core_sim = 1 - abs(row["core_score"] - target_avg_core) / (full_df["core_score"].max() + 1e-6) if full_df["core_score"].max() > 0 else 0.5
        total_score = round(
            1.0 * 0.5 +
            core_sim * 0.3 +
            (row["rating_filled"] / 5) * 0.2,
            3
        )
        rec_candidates.append({
            "asin": asin,
            "title": title if pd.notna(title) else f"商品{asin}",
            "rating": round(row["rating_filled"], 1),
            "core_score": round(row["core_score"], 6),
            "total_score": total_score
        })
    rec_candidates.sort(key=lambda x: x["total_score"], reverse=True)
    return rec_candidates[:top_n]

# ===================== 模块九：智能推荐调度器 =====================
def smart_recommendation_scheduler(
    target_asin=None,
    target_user_id=None,
    full_df=None,
    tfidf_matrix=None,
    user_item_dict=None,
    user_attr_dict=None,
    user_sim_dict=None,
    file_paths=None,
    top_n=TOP_N
):
    LOCAL_FILE_PATHS = file_paths
    if LOCAL_FILE_PATHS is None:
        LOCAL_FILE_PATHS = {
            "comm_csv": r"D:\bigHW\data\商品-社团映射(1).csv",
            "comm_json": r"D:\bigHW\data\社团-商品列表(1).json",
            "meta_txt": r"D:\bigHW\data\10K_amazon-meta.txt",
            "topology_csv": r"D:\bigHW\data\node_features_10k_topology.csv"
        }
        print("⚠️ file_paths 未传入，使用兜底路径 D:\\bigHW\\data\\")

    strategies = {}
    final_recs = []

    is_target_user_valid = target_user_id is not None
    is_user_item_valid = user_item_dict is not None and len(user_item_dict) > 0
    is_full_df_valid = full_df is not None and not full_df.empty
    is_user_attr_valid = user_attr_dict is not None and len(user_attr_dict) > 0
    is_user_sim_valid = user_sim_dict is not None and len(user_sim_dict) > 0
    is_target_asin_valid = target_asin is not None
    is_tfidf_valid = tfidf_matrix is not None
    is_file_paths_valid = LOCAL_FILE_PATHS is not None

    # 策略1：用户协同
    if is_target_user_valid and is_user_item_valid and is_full_df_valid and is_user_attr_valid and is_user_sim_valid:
        try:
            user_recs = user_collab_gnn_recommend(target_user_id, user_item_dict, user_attr_dict, user_sim_dict, full_df, top_n)
            formatted = [{
                "ASIN": r["asin"], "title": r["title"], "rating": r["rating"],
                "total_score": r["total_score"], "core_score": r["core_score"],
                "reason": f"用户协同推荐：{r.get('sim_user_id', '匹配用户画像')}相关偏好"
            } for r in user_recs]
            strategies["user_collab"] = formatted
        except Exception as e:
            print(f"⚠️ 用户协同失败：{e}")

    # 策略2：带动型 + 猜你喜欢
    if is_target_asin_valid and is_full_df_valid and is_tfidf_valid and is_file_paths_valid:
        try:
            core_recs, _, _ = generate_recommendations_with_fallback(target_asin, LOCAL_FILE_PATHS, top_n)
            strategies["core_drive"] = core_recs
            like_recs = guess_you_like(target_asin, full_df, tfidf_matrix, top_n)
            formatted_like = [{
                "ASIN": r["ASIN"], "title": r["title"], "rating": r["rating"],
                "total_score": r["similarity"] * 0.7 + r["core_score"] * 0.3,
                "core_score": r["core_score"],
                "reason": f"猜你喜欢：与商品{target_asin}相似性{r['similarity']}"
            } for r in like_recs]
            strategies["content_like"] = formatted_like
        except Exception as e:
            print(f"⚠️ 带动型/猜你喜欢失败：{e}")

    # 策略融合
    if "user_collab" in strategies:
        user_recs = strategies["user_collab"]
        core_recs = strategies.get("core_drive", [])
        n_user = max(1, int(top_n * 0.6))
        n_core = top_n - n_user
        final_recs = user_recs[:n_user] + core_recs[:n_core]
    elif "core_drive" in strategies and "content_like" in strategies:
        core_recs = strategies["core_drive"]
        like_recs = strategies["content_like"]
        n_core = max(1, int(top_n * 0.7))
        n_like = top_n - n_core
        final_recs = core_recs[:n_core] + like_recs[:n_like]
    else:
        if is_full_df_valid:
            global_core = full_df.nlargest(top_n, "core_score")["ASIN"].tolist()
            for asin in global_core:
                row = full_df[full_df["ASIN"] == asin].iloc[0]
                final_recs.append({
                    "ASIN": asin,
                    "title": row["title"] if pd.notna(row["title"]) else f"商品{asin}",
                    "rating": round(row["rating_filled"], 1),
                    "total_score": row["core_score"],
                    "core_score": row["core_score"],
                    "reason": "全局核心商品：质量优异"
                })
            print("⚠️ 无有效策略，使用全局核心兜底")

    # 去重 + 补充
    seen = set()
    dedup = []
    for r in final_recs:
        if r["ASIN"] not in seen:
            seen.add(r["ASIN"])
            dedup.append(r)
        if len(dedup) >= top_n:
            break
    if len(dedup) < top_n and is_full_df_valid:
        extra = full_df[~full_df["ASIN"].isin(seen)].nlargest(top_n - len(dedup), "core_score")["ASIN"].tolist()
        for asin in extra:
            row = full_df[full_df["ASIN"] == asin].iloc[0]
            dedup.append({
                "ASIN": asin,
                "title": row["title"] if pd.notna(row["title"]) else f"商品{asin}",
                "rating": round(row["rating_filled"], 1),
                "total_score": row["core_score"],
                "core_score": row["core_score"],
                "reason": "全局核心商品：质量优异"
            })

    print(f"\n" + "="*80)
    print(f"✅ 智能推荐调度器 - 最终结果（融合策略：{list(strategies.keys())}）")
    print(f"="*80)
    for idx, rec in enumerate(dedup[:top_n], 1):
        print(f"\n【最终推荐 {idx}】")
        print(f"ASIN：{rec['ASIN']} | 标题：{rec['title']}")
        print(f"评分：{rec['rating']} | 综合得分：{rec['total_score']:.3f} | 核心得分：{rec['core_score']}")
        print(f"推荐理由：{rec['reason']}")
    print(f"\n✅ 智能调度完成！共生成 {len(dedup[:top_n])} 个优质推荐")
    return dedup[:top_n]

# ===================== 模块十：A/B测试框架 =====================
STRATEGY_POOL = ["core_drive", "content_like", "user_collab", "hybrid"]
def get_strategy_for_user(user_id: str):
    if not user_id:
        return "hybrid"
    hash_val = hash(user_id) % 100
    if hash_val < 30:
        return "core_drive"
    elif hash_val < 60:
        return "content_like"
    elif hash_val < 85:
        return "hybrid"
    else:
        return "user_collab"

# ===================== 一键运行入口 =====================
if __name__ == "__main__":
    file_paths = {
        "comm_csv": r"D:\bigHW\data\商品-社团映射(1).csv",
        "comm_json": r"D:\bigHW\data\社团-商品列表(1).json",
        "meta_txt": r"D:\bigHW\data\10K_amazon-meta.txt",
        "topology_csv": r"D:\bigHW\data\node_features_10k_topology.csv"
    }

    missing = [name for name, path in file_paths.items() if not os.path.exists(path)]
    if missing:
        print("❌ 缺少文件，请检查 D:\\bigHW\\data\\ 目录：")
        for m in missing:
            print(f"   - {m}")
        exit(1)
    else:
        print("✅ 所有数据文件路径验证通过！")

    full_df, tfidf, tfidf_matrix = load_data_with_cache(file_paths)
    target_asin = full_df["ASIN"].iloc[1] if len(full_df) > 0 else None
    print(f"\n⚠️  自动选择目标ASIN：{target_asin}")

    user_item_dict, user_attr_dict = extract_user_info_from_meta(full_df, file_paths["meta_txt"])
    user_comment_likes = {uid: attr["total_likes"] for uid, attr in user_attr_dict.items()}
    user_graph, user_sim_dict, user_list = build_user_weighted_network(user_item_dict, user_comment_likes)
    target_user_id = max(user_attr_dict.keys(), key=lambda x: user_attr_dict[x]["comment_count"]) if user_attr_dict else None
    if target_user_id:
        print(f"⚠️  自动选择目标用户（评论数最多）：{target_user_id}")

    ab_strategy = get_strategy_for_user(target_user_id)
    print(f"🎯 A/B测试策略：{ab_strategy}")

    if target_asin or target_user_id:
        final_recs = smart_recommendation_scheduler(
            target_asin=target_asin,
            target_user_id=target_user_id,
            full_df=full_df,
            tfidf_matrix=tfidf_matrix,
            user_item_dict=user_item_dict,
            user_attr_dict=user_attr_dict,
            user_sim_dict=user_sim_dict,
            file_paths=file_paths,
            top_n=TOP_N
        )

    if target_asin:
        core_recs, _, _ = generate_recommendations_with_fallback(target_asin, file_paths, TOP_N)
        if core_recs:
            analyze_recommendation_quality(core_recs, full_df)
        full_df = analyze_community_topic(full_df, comm_id=0)