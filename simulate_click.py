# simulate_click.py 
import requests
import random
import time
import sys
import traceback
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 45
MAX_RETRIES = 3
RETRY_DELAY = 2

def create_session():
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def safe_get(data, key, default="N/A"):
    return data.get(key, data.get("detail", data.get("message", default)))

def wait_for_service_ready(max_wait=180):
    print(" 正在检测推荐服务状态...")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            resp = requests.get(f"{BASE_URL}/ready", timeout=5)
            if resp.status_code == 200 and resp.json().get("status") == "ready":
                print("✅ 服务已完全初始化，准备就绪！")
                return True
            else:
                status = resp.json().get("status", "unknown")
                print(f" 服务状态：{status}...")
        except Exception as e:
            print(f" 服务未响应：{e}")
        time.sleep(RETRY_DELAY)
    print(f"❌ 等待超时（{max_wait}秒）！")
    return False

def request_with_retry(method, url, **kwargs):
    session = create_session()
    for i in range(1, MAX_RETRIES + 1):
        try:
            resp = session.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as e:
            print(f"⚠️  请求失败（第 {i}/{MAX_RETRIES} 次）：{e}")
            if i < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise
    return None

def test_recommend_and_click():
    print("=" * 70)
    print(" 启动推荐系统端到端模拟测试")
    print("=" * 70)

    if not wait_for_service_ready():
        return False

    user_id = f"test_user_{int(time.time())}"
    target_asin = "827229534"

    # 1. 获取推荐
    print("\n 正在请求商品推荐...")
    try:
        resp = request_with_retry("POST", f"{BASE_URL}/recommend", json={"user_id": user_id, "target_asin": target_asin, "top_n": 8})
        res = resp.json()
        if res.get("code") != 200:
            print(f"❌ 推荐失败：{safe_get(res, 'message')}")
            return False
        recs = res["data"]["recommendations"]
        print(f"✅ 推荐成功！商品数：{len(recs)}")
    except Exception as e:
        print(f"❌ 推荐请求异常：{e}")
        return False

    # 2. 模拟点击
    clicked = []
    print("\n🖱️  正在模拟点击行为...")
    for r in recs[:3]:
        if random.random() < 0.7:
            asin = r["ASIN"]
            try:
                click_resp = request_with_retry("POST", f"{BASE_URL}/log/click", json={"user_id": user_id, "asin": asin})
                print(f"   点击 {asin} → {safe_get(click_resp.json(), 'message')}")
                clicked.append(asin)
            except Exception as e:
                print(f"  ⚠️  点击 {asin} 失败：{e}")

    # 3. 模拟转化
    if clicked:
        print("\n 正在模拟转化行为...")
        time.sleep(0.5)
        for asin in clicked:
            if random.random() < 0.4:
                try:
                    conv_resp = request_with_retry("POST", f"{BASE_URL}/log/convert", json={"user_id": user_id, "asin": asin})
                    print(f"  转化 {asin} → {safe_get(conv_resp.json(), 'message')}")
                except Exception as e:
                    print(f"  ⚠️  转化 {asin} 失败：{e}")

    # 4. A/B 统计（修复语法错误）
    print("\n 正在获取 A/B 测试统计...")
    try:
        stats_resp = request_with_retry("GET", f"{BASE_URL}/ab/stats")
        stats = stats_resp.json()
        data = stats.get("data", [])
        if not data: 
            print("  ⚠️  暂无统计数据")
        else:
            print("\n📈 A/B 测试效果统计：")
            for s in data:  
                print(f"  • {s.get('strategy', 'N/A'):<14} | 曝光{s.get('exposures', 0):>4}次 | CTR={s.get('ctr', 0):>6.1%}")
    except Exception as e:
        print(f"❌ A/B 统计异常：{e}")
        return False

    print("\n 模拟测试完成！")
    return True

if __name__ == "__main__":
    success = test_recommend_and_click()
    sys.exit(0 if success else 1)