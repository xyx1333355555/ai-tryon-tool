import os
import streamlit as st
import requests
import time
import io
import traceback
import urllib3
import base64
import uuid
import cv2
import numpy as np

_TINYFACE_CACHE_DIR = os.path.join(os.getcwd(), 'models_cache')
os.makedirs(_TINYFACE_CACHE_DIR, exist_ok=True)
os.environ['FACE_DETECTOR_MODEL'] = os.path.join(_TINYFACE_CACHE_DIR, 'scrfd_2.5g.onnx')
os.environ['FACE_EMBEDDER_MODEL'] = os.path.join(_TINYFACE_CACHE_DIR, 'arcface_w600k_r50.onnx')
os.environ['FACE_SWAPPER_MODEL'] = os.path.join(_TINYFACE_CACHE_DIR, 'inswapper_128_fp16.onnx')
os.environ['FACE_ENHANCER_MODEL'] = os.path.join(_TINYFACE_CACHE_DIR, 'gfpgan_1.4.onnx')
from PIL import Image

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="AI 虚拟试衣间", page_icon="👗", layout="wide")

DASHSCOPE_BASE = os.environ.get("DASHSCOPE_BASE", "https://dashscope.aliyuncs.com")
CREATE_TASK_URL = f"{DASHSCOPE_BASE}/api/v1/services/aigc/image2image/image-synthesis"
TEXT2IMAGE_URL = f"{DASHSCOPE_BASE}/api/v1/services/aigc/text2image/image-synthesis"
IMAGE_EDIT_URL = f"{DASHSCOPE_BASE}/api/v1/services/aigc/multimodal-generation/generation"
QUERY_TASK_URL = f"{DASHSCOPE_BASE}/api/v1/tasks/{{task_id}}"

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    max_retries=urllib3.Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
)
session.mount('https://', adapter)
session.mount('http://', adapter)


def image_to_base64(file_obj, max_size=55000):
    """将图片压缩转换为 Base64，严格控制大小"""
    img = Image.open(file_obj)
    if img.mode in ("RGBA", "P", "CMYK"):
        img = img.convert("RGB")

    # 对于全身照，优先保持高度比例
    w, h = img.size
    target_h = 1024
    if h > target_h:
        target_w = int(w * target_h / h)
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    qualities = [85, 75, 65, 55, 45, 35]

    for quality in qualities:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        buf.seek(0)
        base64_str = base64.b64encode(buf.read()).decode("utf-8")

        if len(base64_str) < max_size:
            return base64_str

    # 最终保底方案
    img.thumbnail((512, 1024), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=40, optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def analyze_fashion_style(top_bytes=None, bottom_bytes=None):
    """分析穿搭风格，生成搞笑的时尚人格画像"""
    import random

    # 颜色分析（简单的RGB平均值）
    def get_dominant_color(img_bytes):
        try:
            img = Image.open(io.BytesIO(img_bytes))
            img = img.convert('RGB')
            img = img.resize((50, 50))
            pixels = list(img.getdata())
            r = sum(p[0] for p in pixels) // len(pixels)
            g = sum(p[1] for p in pixels) // len(pixels)
            b = sum(p[2] for p in pixels) // len(pixels)
            return (r, g, b)
        except:
            return (128, 128, 128)

    def rgb_to_color_name(r, g, b):
        colors = {
            '黑色': (0, 0, 0), '白色': (255, 255, 255), '红色': (255, 0, 0),
            '绿色': (0, 128, 0), '蓝色': (0, 0, 255), '黄色': (255, 255, 0),
            '紫色': (128, 0, 128), '粉色': (255, 192, 203), '橙色': (255, 165, 0),
            '灰色': (128, 128, 128), '棕色': (165, 42, 42), '米色': (245, 245, 220),
            '藏青': (0, 0, 128), '酒红': (128, 0, 0), '卡其': (195, 176, 145),
        }
        min_dist = float('inf')
        closest = '神秘色'
        for name, (cr, cg, cb) in colors.items():
            dist = ((r-cr)**2 + (g-cg)**2 + (b-cb)**2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest = name
        return closest

    # 分析上衣颜色
    top_color = '未知'
    if top_bytes:
        r, g, b = get_dominant_color(top_bytes)
        top_color = rgb_to_color_name(r, g, b)

    # 分析下装颜色
    bottom_color = '未知'
    if bottom_bytes:
        r, g, b = get_dominant_color(bottom_bytes)
        bottom_color = rgb_to_color_name(r, g, b)

    # 时尚人格类型定义 - 基于2024-2025网络热梗
    personalities = {
        'G-O-R-P': {
            'name': 'Gorpcore Outdoor Research Person',
            'title': '山系户外研究猿',
            'emoji': '🏔️',
            'traits': ['冲锋衣是皮肤', '始祖鸟是信仰', '露营是周末唯一活动', '城市里的山野人'],
            'slogan': '我不是在上班，我是在等下班去爬山',
            'compatibility': '与登山杖、露营灯、速干面料绝配',
            'warning': '你的冲锋衣比你的社交活动还多',
            'catchphrase': '山不见我，我自去见山',
        },
        'B-R-A-T': {
            'name': 'Brat Summer Aesthetic Trend',
            'title': 'Brat夏日美学先锋',
            'emoji': '💚',
            'traits': ['荧光绿是本命色', '派对永远不散场', '365天都是夏天', 'Club是我第二个家'],
            'slogan': '我不是在派对，就是在去派对的路上',
            'compatibility': '与霓虹灯、电子音乐、凌晨三点的出租车绝配',
            'warning': '你的肝脏可能比你的衣橱更需要休息',
            'catchphrase': '365 party girl',
        },
        'C-H-I-L-L': {
            'name': 'Chill Guy Lifestyle Legend',
            'title': 'Chill Guy松弛感大师',
            'emoji': '🐕',
            'traits': ['灰色毛衣是战袍', '双手插兜走天下', '什么都不在乎', '松弛感拉满'],
            'slogan': '我只是个Chill Guy，别问我意见',
            'compatibility': '与灰色系、宽松版型、无所谓态度绝配',
            'warning': '你的松弛可能已经松到裤子要掉了',
            'catchphrase': 'I am just a chill guy',
        },
        'L-O-S-E-R': {
            'name': 'Loser Core Aesthetic Rebel',
            'title': 'Loser Core反骨美学',
            'emoji': '🤓',
            'traits': [' nerd是最高赞美', '反精致是态度', '邋遢是艺术', '不fit in是fit in'],
            'slogan': '我不是不会穿，我是不想会穿',
            'compatibility': '与 oversized、皱巴巴、不合身绝配',
            'warning': '你的"故意穿丑"可能已经变成"真的丑"',
            'catchphrase': 'Being a loser is the new cool',
        },
        'M-O-B-W': {
            'name': 'Mob Wife Aesthetic Boss',
            'title': 'Mob Wife大佬女人',
            'emoji': '🐆',
            'traits': ['豹纹是基本款', '皮草是日常', '金链子是配饰', '走路带风'],
            'slogan': '我不是在穿皮草，我是在穿态度',
            'compatibility': '与高跟鞋、大红唇、霸气眼神绝配',
            'warning': '你的气场可能把邻居吓到了',
            'catchphrase': 'I am the wife, I am the mob',
        },
        'C-O-Q-U-E': {
            'name': 'Coquette Bow Obsessed',
            'title': 'Coquette蝴蝶结精',
            'emoji': '🎀',
            'traits': ['蝴蝶结是本体', '粉色是空气', '蕾丝是皮肤', '甜美是武器'],
            'slogan': '我不是在装可爱，我是真可爱',
            'compatibility': '与丝带、蓬蓬裙、珍珠发夹绝配',
            'warning': '你的蝴蝶结可能比你的头发还多',
            'catchphrase': 'Bow down to the bow',
        },
        'O-L-D-M': {
            'name': 'Old Money Quiet Luxury',
            'title': '老钱静奢风贵族',
            'emoji': '🎩',
            'traits': ['logo是禁忌', '质感是唯一标准', '低调是最高调', '看起来很有钱但不说'],
            'slogan': '我不穿品牌，品牌穿我',
            'compatibility': '与羊绒、亚麻、无logo绝配',
            'warning': '你的"看起来贵"可能只是"看起来旧"',
            'catchphrase': 'Money talks, wealth whispers',
        },
        'Y-2-K-R': {
            'name': 'Y2K Revival Retro Queen',
            'title': 'Y2K复古千禧女王',
            'emoji': '✨',
            'traits': ['低腰裤是信仰', '亮片是日常', '翻盖手机是配饰', '2000年从未离开'],
            'slogan': '我不是复古，我是从未过时',
            'compatibility': '与蝴蝶元素、金属色、厚底鞋绝配',
            'warning': '你的低腰裤可能让肚脐感冒了',
            'catchphrase': 'What is dead may never die',
        },
        'P-R-E-P-P': {
            'name': 'Preppy Academia Prince',
            'title': '预科风学术王子/公主',
            'emoji': '📚',
            'traits': ['格子衬衫是校服', '毛衣搭肩是标配', '图书馆是T台', '常春藤是梦想'],
            'slogan': '我不是在学习，我是在走秀',
            'compatibility': '与乐福鞋、领带、精装书绝配',
            'warning': '你的书可能比你的衣服还新',
            'catchphrase': 'Ivy League state of mind',
        },
        'N-O-R-M-C': {
            'name': 'Normcore Basic Legend',
            'title': 'Normcore基础款传奇',
            'emoji': '👕',
            'traits': ['白T是礼服', '牛仔裤是正装', '帆布鞋是高跟鞋', '普通是最高级'],
            'slogan': '我不是没风格，我的风格就是没风格',
            'compatibility': '与优衣库、基础款、不费力绝配',
            'warning': '你的"不费力"可能只是"没努力"',
            'catchphrase': 'Less is more, basic is best',
        },
    }

    # 根据颜色组合选择人格 - 更智能的匹配
    if top_color in ['黑色', '白色', '灰色'] and bottom_color in ['黑色', '白色', '灰色']:
        personality_key = random.choice(['C-H-I-L-L', 'N-O-R-M-C', 'O-L-D-M'])
    elif top_color in ['红色', '橙色'] or bottom_color in ['红色', '橙色']:
        personality_key = random.choice(['M-O-B-W', 'B-R-A-T'])
    elif top_color in ['黄色', '金色'] or bottom_color in ['黄色', '金色']:
        personality_key = 'Y-2-K-R'
    elif top_color in ['蓝色', '藏青'] or bottom_color in ['蓝色', '藏青']:
        personality_key = random.choice(['P-R-E-P-P', 'G-O-R-P'])
    elif top_color in ['绿色', '卡其'] or bottom_color in ['绿色', '卡其']:
        personality_key = random.choice(['G-O-R-P', 'C-H-I-L-L'])
    elif top_color in ['粉色', '紫色'] or bottom_color in ['粉色', '紫色']:
        personality_key = random.choice(['C-O-Q-U-E', 'B-R-A-T'])
    elif top_color in ['棕色', '米色'] or bottom_color in ['棕色', '米色']:
        personality_key = random.choice(['O-L-D-M', 'C-H-I-L-L'])
    else:
        personality_key = random.choice(list(personalities.keys()))

    # 如果只有一件，根据单件颜色选择
    if not top_bytes and not bottom_bytes:
        personality_key = random.choice(list(personalities.keys()))

    personality = personalities[personality_key]

    # 生成时尚建议 - 更有梗更有趣
    fashion_tips = [
        f"今天你的{top_color}上衣正在向全世界发射信号：'{personality['slogan']}'",
        f"搭配{top_color}和{bottom_color}的你，正在重新定义色彩心理学",
        f"警告：你的穿搭可能导致路人频繁回头，请做好心理准备",
        f"时尚侦探分析：你的衣橱里有70%是{top_color}系，这是有预谋的",
        f"你的穿搭风格已经超越了'好看'，进入了'{personality['title']}'的境界",
        f"如果穿搭是一门课，你已经挂科了——因为你太超前了",
    ]

    return {
        'type': personality_key,
        'name': personality['name'],
        'title': personality['title'],
        'emoji': personality.get('emoji', '👔'),
        'traits': personality['traits'],
        'slogan': personality['slogan'],
        'compatibility': personality['compatibility'],
        'warning': personality['warning'],
        'catchphrase': personality.get('catchphrase', ''),
        'top_color': top_color,
        'bottom_color': bottom_color,
        'tips': random.sample(fashion_tips, min(3, len(fashion_tips))),
    }


def upload_image_to_dashscope(api_key, image_bytes, model_name="wanx2.1-t2i-plus"):
    """上传图片到百炼临时存储，获取 oss:// URL（有效期48小时）"""
    filename = f"ref_{uuid.uuid4().hex[:8]}.jpg"

    # 1. 获取上传凭证
    policy_url = "https://dashscope.aliyuncs.com/api/v1/uploads"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    params = {
        "action": "getPolicy",
        "model": model_name
    }

    resp = requests.get(policy_url, headers=headers, params=params, timeout=30, verify=False)
    if resp.status_code != 200:
        raise Exception(f"获取上传凭证失败: {resp.text}")

    policy_data = resp.json()["data"]

    # 2. 上传文件到 OSS
    key = f"{policy_data['upload_dir']}/{filename}"
    files = {
        'OSSAccessKeyId': (None, policy_data['oss_access_key_id']),
        'Signature': (None, policy_data['signature']),
        'policy': (None, policy_data['policy']),
        'x-oss-object-acl': (None, policy_data['x_oss_object_acl']),
        'x-oss-forbid-overwrite': (None, policy_data['x_oss_forbid_overwrite']),
        'key': (None, key),
        'success_action_status': (None, '200'),
        'file': (filename, io.BytesIO(image_bytes))
    }

    upload_resp = requests.post(policy_data['upload_host'], files=files, timeout=60, verify=False)
    if upload_resp.status_code != 200:
        raise Exception(f"上传文件失败: {upload_resp.text}")

    # 3. 返回 oss:// URL
    return f"oss://{key}"


def submit_image_to_image(api_key, image_base64):
    """使用 wan2.6-image 图像编辑模式，基于原图生成全身照（图生图）"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": "wan2.6-image",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": "Based on this person's photo, generate a complete full-body portrait of the exact same person standing naturally with arms relaxed at sides, feet fully visible. Keep the identical face, hairstyle, ears, skin tone, and body proportions. The person must be shown from head to toe with complete legs and feet visible in the frame. Full body shot, entire person visible. Professional photo quality, solid light blue background."
                        },
                        {
                            "image": f"data:image/jpeg;base64,{image_base64}"
                        }
                    ]
                }
            ]
        },
        "parameters": {
            "size": "768*1344",
            "n": 1,
            "enable_interleave": False,
            "prompt_extend": True
        }
    }

    resp = requests.post(IMAGE_EDIT_URL, headers=headers, json=payload, timeout=120, verify=False)
    data = resp.json()

    if resp.status_code != 200:
        error_msg = data.get("message", str(data))
        raise Exception(f"图生图请求失败: {error_msg}")

    choices = data.get("output", {}).get("choices", [])
    if not choices:
        raise Exception(f"图生图未返回结果: {data}")

    images = []
    for choice in choices:
        for content in choice.get("message", {}).get("content", []):
            if content.get("type") == "image" and content.get("image"):
                images.append(content["image"])

    if not images:
        raise Exception(f"图生图未返回图片: {data}")

    return images[0]


def submit_try_on(api_key, person_url, top_url=None, bottom_url=None, model="aitryon", resolution=-1, restore_face=True):
    """提交试衣任务"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "X-DashScope-Async": "enable",
    }

    payload = {
        "model": model,
        "input": {"person_image_url": person_url},
        "parameters": {"resolution": resolution, "restore_face": restore_face},
    }
    if top_url:
        payload["input"]["top_garment_url"] = top_url
    if bottom_url:
        payload["input"]["bottom_garment_url"] = bottom_url

    resp = session.post(CREATE_TASK_URL, headers=headers, json=payload, timeout=60)
    data = resp.json()

    if resp.status_code != 200 or "output" not in data:
        error_msg = data.get("message", str(data))
        raise Exception(f"创建试衣任务失败: {error_msg}")

    return data["output"]["task_id"]


def wait_for_result(api_key, task_id):
    """等待任务完成并返回结果"""
    headers = {"Authorization": f"Bearer {api_key}"}
    max_wait = 180
    started = time.time()

    while time.time() - started < max_wait:
        resp = session.get(QUERY_TASK_URL.format(task_id=task_id), headers=headers, timeout=30)
        data = resp.json()
        output = data.get("output", {})
        status = output.get("task_status", "UNKNOWN")

        if status == "SUCCEEDED":
            # 文生图/扩图 API 返回 results[].url，试衣 API 返回 image_url
            results = output.get("results", [])
            if results and len(results) > 0:
                return results[0].get("url", "").strip()
            result_url = output.get("output_image_url") or output.get("image_url")
            if result_url:
                return result_url.strip()
            raise Exception("未获取到结果图片 URL")
        elif status == "FAILED":
            msg = output.get("message", "未知错误")
            raise Exception(f"任务失败: {msg}")
        elif status == "CANCELED":
            raise Exception("任务已取消")

        time.sleep(3)

    raise Exception("任务超时（超过 3 分钟）")


def simple_face_swap(source_img_bytes, target_img_bytes):
    """使用 TinyFace (MagicMirror核心) 进行专业 AI 换脸"""
    import sys
    from tinyface import TinyFace
    
    cache_dir = os.path.join(os.getcwd(), 'models_cache')
    
    required_models = {
        'scrfd_2.5g.onnx': os.environ.get('FACE_DETECTOR_MODEL'),
        'arcface_w600k_r50.onnx': os.environ.get('FACE_EMBEDDER_MODEL'),
        'inswapper_128_fp16.onnx': os.environ.get('FACE_SWAPPER_MODEL'),
        'gfpgan_1.4.onnx': os.environ.get('FACE_ENHANCER_MODEL'),
    }
    
    missing = [name for name, path in required_models.items() if not path or not os.path.exists(path)]
    if missing:
        print(f"[换脸] 错误: 模型文件缺失: {', '.join(missing)}", flush=True)
        print(f"[换脸] 模型目录: {cache_dir}", flush=True)
        for name, path in required_models.items():
            print(f"  - {name}: {'存在' if path and os.path.exists(path) else '缺失'}", flush=True)
        return None
    
    print("[换脸] 开始初始化 TinyFace AI 换脸...", flush=True)
    sys.stdout.flush()
    
    source_img = cv2.imdecode(np.frombuffer(source_img_bytes, np.uint8), cv2.IMREAD_COLOR)
    target_img = cv2.imdecode(np.frombuffer(target_img_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    if source_img is None or target_img is None:
        print("[换脸] 图片加载失败", flush=True)
        return None
    
    print("[换脸] 图片加载成功", flush=True)
    
    if not hasattr(simple_face_swap, '_tinyface'):
        print("[换脸] 首次运行，正在创建 TinyFace 实例...", flush=True)
        try:
            simple_face_swap._tinyface = TinyFace()
            print("[换脸] 正在加载本地 AI 模型到内存（无需下载）...", flush=True)
            simple_face_swap._tinyface.prepare()
            print("[换脸] TinyFace AI 模型准备完成！", flush=True)
        except Exception as e:
            print(f"[换脸] TinyFace 模型加载失败: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
    
    tinyface = simple_face_swap._tinyface
    
    print("[换脸] 正在检测人脸...", flush=True)
    source_faces = tinyface.get_many_faces(source_img)
    target_faces = tinyface.get_many_faces(target_img)
    
    if len(source_faces) == 0 or len(target_faces) == 0:
        print(f"[换脸] 未检测到人脸 (目标: {len(target_faces)}, 源: {len(source_faces)})", flush=True)
        return None
    
    print(f"[换脸] 检测到人脸 (目标: {len(target_faces)}, 源: {len(source_faces)})", flush=True)
    
    source_face = max(source_faces, key=lambda f: f.bounding_box[2] * f.bounding_box[3])
    target_face = max(target_faces, key=lambda f: f.bounding_box[2] * f.bounding_box[3])
    
    print("[换脸] 正在执行 AI 换脸（MagicMirror 方式）...", flush=True)
    result = tinyface.swap_face(
        vision_frame=target_img,
        reference_face=target_face,
        destination_face=source_face
    )
    
    print("[换脸] 换脸完成！", flush=True)
    
    _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buffer.tobytes()


def _insightface_fallback(source_img_bytes, target_img_bytes):
    """备用方案：使用 InsightFace"""
    try:
        import insightface
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model
        
        source_img = cv2.imdecode(np.frombuffer(source_img_bytes, np.uint8), cv2.IMREAD_COLOR)
        target_img = cv2.imdecode(np.frombuffer(target_img_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if source_img is None or target_img is None:
            return None
        
        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        source_faces = app.get(source_img)
        target_faces = app.get(target_img)
        
        if len(source_faces) == 0 or len(target_faces) == 0:
            return None
        
        source_face = max(source_faces, key=lambda f: f.bbox[2] * f.bbox[3])
        target_face = max(target_faces, key=lambda f: f.bbox[2] * f.bbox[3])
        
        swapper = get_model('w600k_r50.onnx', download=True, download_zip=True)
        result = swapper.get(target_img, target_face, source_face, paste_back=True)
        
        _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return buffer.tobytes()
        
    except Exception as e:
        print(f"InsightFace 也失败了: {e}")
        return None


def process_half_body_to_full(api_key, image_file, progress_placeholder):
    """使用 wan2.6-image 图生图：基于原图生成全身照（保持原人物特征）"""
    progress_placeholder.info("🔄 阶段 1/2：正在基于原图生成全身照（图生图）...")

    image_file.seek(0)
    image_base64 = image_to_base64(image_file)
    progress_placeholder.info("📤 图片处理完成，正在调用图生图 API...")

    result_url = submit_image_to_image(api_key, image_base64)
    progress_placeholder.success("✅ 全身照生成完成！（图生图，保持原人物特征）")

    return result_url


st.title("👗 AI 虚拟试衣间")
st.markdown("阿里云百炼 AI 试衣 API · OutfitAnyone 模型 · [免费注册](https://bailian.console.aliyun.com)")

with st.sidebar:
    st.header("⚙️ 设置")

    api_key = st.text_input(
        "阿里云百炼 API Key",
        type="password",
        placeholder="sk-xxxxxxxxxxxxxxxx",
        help="[bailian.console.aliyun.com](https://bailian.console.aliyun.com) → API-KEY 管理",
    )

    st.divider()

    model_choice = st.selectbox(
        "模型",
        options=["aitryon", "aitryon-plus"],
        format_func=lambda x: (
            "AI试衣-基础版 (0.20 元/张, ~15s)"
            if x == "aitryon"
            else "AI试衣-Plus版 (0.50 元/张, ~30s, 效果更好)"
        ),
    )

    st.divider()

    auto_extend = st.checkbox(
        "🩳 半身照自动补全为全身照",
        value=True,
        help="使用通义万相图生图技术，基于原图生成全身照（保持原人物特征，额外消耗约 0.04 元/张）"
    )

    st.caption(f"💰 试衣: {model_choice == 'aitryon' and '0.20' or '0.50'} 元/张")
    if auto_extend:
        st.caption("💰 图生图全身照: ~0.04 元/张 (半身照补全)")
    st.caption("🎁 新用户免费 400 张额度")

st.markdown("### ① 上传图片")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 人物照片")
    person_file = st.file_uploader(
        "上传人物照片（半身照或全身照均可）",
        type=["jpg", "jpeg", "png", "webp", "bmp", "heic"],
        key="person"
    )
    if person_file is not None:
        person_bytes = person_file.getvalue()
        st.image(person_bytes, use_container_width=True)

with col2:
    st.subheader("👔 衣服图片")
    tab1, tab2 = st.tabs(["上装/连衣裙", "下装（可选）"])
    with tab1:
        top_file = st.file_uploader(
            "上装或连衣裙",
            type=["jpg", "jpeg", "png", "webp", "bmp", "heic"],
            key="top"
        )
        if top_file is not None:
            top_bytes = top_file.getvalue()
            st.image(top_bytes, use_container_width=True)
    with tab2:
        bottom_file = st.file_uploader(
            "下装（裤子/裙子）",
            type=["jpg", "jpeg", "png", "webp", "bmp", "heic"],
            key="bottom"
        )
        if bottom_file is not None:
            bottom_bytes = bottom_file.getvalue()
            st.image(bottom_bytes, use_container_width=True)

with st.expander("⚙️ 高级参数"):
    col_a, col_b = st.columns(2)
    with col_a:
        resolution = st.selectbox(
            "输出分辨率",
            options=[1280, 1024, -1],
            format_func=lambda x: {-1: "保持原图", 1024: "1024 (576×1024)", 1280: "1280 (720×1280) - 全身照推荐"}[x],
        )
    with col_b:
        restore_face = st.checkbox("保留原人脸", value=True)

if auto_extend:
    st.info("💡 已开启「半身照自动补全」：将使用通义万相图生图技术，基于原图生成全身照（保持原人物发型、五官、体型等特征）")
else:
    st.info("💡 提示：建议上传全身人物照片以获得最佳试衣效果")

if st.button("✨ 开始试穿", type="primary", use_container_width=True):
    if not api_key:
        st.error("请在左侧输入阿里云百炼 API Key")
        st.stop()

    if person_file is None:
        st.error("请上传人物照片")
        st.stop()

    if top_file is None and bottom_file is None:
        st.error("请至少上传一件上装或下装")
        st.stop()

    progress_area = st.empty()

    try:
        person_url = None

        if auto_extend:
            person_url = process_half_body_to_full(api_key, person_file, progress_area)
            if person_url:
                st.session_state['full_body_url'] = person_url
                full_body_resp = requests.get(person_url, timeout=30, verify=False)
                if full_body_resp.status_code == 200:
                    st.session_state['full_body_bytes'] = full_body_resp.content
            else:
                raise Exception("未能获取到图生图结果")
        else:
            progress_area.info("🔄 正在处理人物照片...")
            person_file.seek(0)
            person_b64 = image_to_base64(person_file)
            person_url = f"data:image/jpeg;base64,{person_b64}"
            progress_area.success(f"✅ 人物照片处理完成")

        top_url = None
        bottom_url = None

        if top_file is not None:
            with st.spinner("正在处理上装照片..."):
                top_file.seek(0)
                top_b64 = image_to_base64(top_file)
                top_url = f"data:image/jpeg;base64,{top_b64}"
                st.success(f"上装照片处理完成")

        if bottom_file is not None:
            with st.spinner("正在处理下装照片..."):
                bottom_file.seek(0)
                bottom_b64 = image_to_base64(bottom_file)
                bottom_url = f"data:image/jpeg;base64,{bottom_b64}"
                st.success(f"下装照片处理完成")

        progress_area.info("🔄 阶段 2/2：正在进行 AI 试衣...")
        task_id = submit_try_on(
            api_key,
            person_url,
            top_url,
            bottom_url,
            model=model_choice,
            resolution=resolution,
            restore_face=restore_face
        )
        progress_area.info(f"📤 试衣任务已提交，ID: {task_id[:8]}...")

        result_url = wait_for_result(api_key, task_id)

        if result_url:
            st.session_state['tryon_url'] = result_url
            tryon_resp = requests.get(result_url, timeout=30, verify=False)
            if tryon_resp.status_code == 200:
                st.session_state['tryon_bytes'] = tryon_resp.content
            progress_area.empty()
            st.success("✅ 试衣完成！")
        else:
            st.error("未获取到结果图片")

    except Exception as e:
        progress_area.empty()
        st.error(f"出错了: {str(e)}")
        with st.expander("查看详细错误信息"):
            st.code(traceback.format_exc())

if st.session_state.get('full_body_url'):
    st.subheader("🦵 图生图生成的全身照")
    st.image(st.session_state['full_body_url'], caption="通义万相图生图生成的全身照（保持原人物特征）", use_container_width=True)
    if st.session_state.get('full_body_bytes'):
        st.download_button(
            label="⬇️ 下载全身照",
            data=st.session_state['full_body_bytes'],
            file_name="full_body.jpg",
            mime="image/jpeg",
            key="download_fullbody"
        )
    st.divider()

if st.session_state.get('tryon_url'):
    st.subheader("👗 试衣结果")
    st.image(st.session_state['tryon_url'], caption="试穿效果", use_container_width=True)
    if st.session_state.get('tryon_bytes'):
        st.download_button(
            label="⬇️ 下载试衣结果",
            data=st.session_state['tryon_bytes'],
            file_name="tryon_result.jpg",
            mime="image/jpeg",
            key="download_tryon"
        )
    else:
        st.markdown(f"[点击下载图片]({st.session_state['tryon_url']})")

    # 时尚人格画像分析
    st.divider()
    with st.container():
        st.subheader("🎭 你的时尚人格画像")

        # 获取上传的衣服图片bytes
        top_bytes = None
        bottom_bytes = None
        if top_file is not None:
            top_file.seek(0)
            top_bytes = top_file.getvalue()
        if bottom_file is not None:
            bottom_file.seek(0)
            bottom_bytes = bottom_file.getvalue()

        style_result = analyze_fashion_style(top_bytes, bottom_bytes)

        # 展示结果 - 更有趣的展示方式
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 25px; border-radius: 20px; text-align: center; color: white; margin-bottom: 20px;'>
            <div style='font-size: 60px; margin-bottom: 10px;'>{style_result['emoji']}</div>
            <h1 style='font-size: 42px; margin: 0; font-weight: bold;'>{style_result['type']}</h1>
            <p style='font-size: 16px; margin-top: 8px; opacity: 0.9;'>你的时尚人格类型</p>
        </div>
        """, unsafe_allow_html=True)

        col_info, col_traits = st.columns([2, 1])
        with col_info:
            st.markdown(f"### {style_result['name']}")
            st.markdown(f"**{style_result['title']}**")
            st.markdown(f"\n🎯 **人格特质：**")
            for trait in style_result['traits']:
                st.markdown(f"- {trait}")

        with col_traits:
            st.markdown(f"\n📊 **色彩分析：**")
            st.markdown(f"- 上装：{style_result['top_color']}")
            st.markdown(f"- 下装：{style_result['bottom_color']}")
            st.markdown(f"\n🏷️ **标签：**")
            st.markdown(f"`{style_result['type']}` `{style_result['title'][:6]}...`")

        st.markdown(f"\n💬 **你的时尚宣言：**")
        st.info(f"*{style_result['slogan']}*")

        if style_result.get('catchphrase'):
            st.markdown(f"\n🔥 **口头禅：**")
            st.success(f"_{style_result['catchphrase']}_")

        st.markdown(f"\n🔮 **AI穿搭分析师点评：**")
        for tip in style_result['tips']:
            st.markdown(f"- {tip}")

        col_compat, col_warn = st.columns(2)
        with col_compat:
            st.markdown(f"\n✨ **最佳搭配：**")
            st.success(style_result['compatibility'])
        with col_warn:
            st.markdown(f"\n⚠️ **时尚警告：**")
            st.warning(style_result['warning'])

        st.divider()
        st.caption("📊 分析基于2024-2025网络热梗 + 服装颜色AI计算得出，仅供娱乐~")

st.divider()
st.subheader("🔄 AI 换脸工具")
st.markdown("把 A 照片的脸，换到 B 照片上")

swap_col1, swap_col2 = st.columns(2)
with swap_col1:
    st.markdown("**A：提供脸的照片**（你想用谁的脸）")
    swap_source = st.file_uploader(
        "上传带脸的照片",
        type=["jpg", "jpeg", "png", "webp", "bmp", "heic"],
        key="swap_source"
    )
    if swap_source is not None:
        st.image(swap_source.getvalue(), use_container_width=True)

with swap_col2:
    st.markdown("**B：被换脸的照片**（你想把脸换到哪张照片上）")
    swap_target = st.file_uploader(
        "上传目标照片",
        type=["jpg", "jpeg", "png", "webp", "bmp", "heic"],
        key="swap_target"
    )
    if swap_target is not None:
        st.image(swap_target.getvalue(), use_container_width=True)

if st.button("🔄 开始换脸", use_container_width=True):
    if swap_source is None or swap_target is None:
        st.error("请上传人脸源照片和目标照片")
        st.stop()

    with st.spinner("正在进行 AI 换脸..."):
        try:
            swap_source.seek(0)
            source_bytes = swap_source.read()
            swap_target.seek(0)
            target_bytes = swap_target.read()

            result_bytes = simple_face_swap(source_bytes, target_bytes)

            if result_bytes:
                st.session_state['faceswap_bytes'] = result_bytes
                st.success("✅ 换脸完成！")
            else:
                st.error("换脸失败：未检测到人脸，请确保两张照片都包含清晰的人脸")
        except Exception as e:
            st.error(f"换脸出错: {str(e)}")
            with st.expander("查看详细错误信息"):
                st.code(traceback.format_exc())

if st.session_state.get('faceswap_bytes'):
    st.image(st.session_state['faceswap_bytes'], caption="换脸结果", use_container_width=True)
    st.download_button(
        label="⬇️ 下载换脸结果",
        data=st.session_state['faceswap_bytes'],
        file_name="face_swap_result.jpg",
        mime="image/jpeg",
        key="download_faceswap"
    )

with st.expander("📖 使用说明"):
    st.markdown("""
    ## 使用步骤
    1. 在左侧输入阿里云百炼 API Key（[点击获取](https://bailian.console.aliyun.com)）
    2. 选择模型版本（基础版 0.20 元/张 或 Plus 版 0.50 元/张）
    3. 勾选「半身照自动补全为全身照」（默认开启，推荐）
    4. 上传人物照片（半身照或全身照均可）
    5. 上传衣服图片（上装或下装至少一件）
    6. 点击「开始试穿」等待结果

    ## 半身照补全原理
    开启后，系统会自动执行两阶段处理：
    - **阶段 1**：使用通义万相 `wan2.6-image` 图生图模型，基于原半身照生成全身照（保持原人物发型、五官、体型等特征）
    - **阶段 2**：使用 OutfitAnyone 模型进行试衣

    ## AI 换脸工具
    页面下方的独立功能，可随时使用：
    - 上传人脸源照片 + 目标照片 → 一键换脸
    - 基于TinyFace（MagicMirror核心）技术
    - 适用于试衣后人脸需要精细调整的场景

    ## 费用说明
    - 图生图全身照：约 0.04 元/张（半身照补全）
    - 试衣：0.20 元/张（基础版）或 0.50 元/张（Plus 版）
    - AI 换脸：免费（本地运行，无需 API）
    - 如果上传的是全身照，关闭「自动补全」可节省图生图费用

    ## 图片要求
    - **人物照**：单人、正面、光照良好、背景简洁
    - **衣服照**：单件平铺、无遮挡褶皱、背景干净、占画面主体
    - **格式**：JPG/PNG/BMP/HEIC，5KB-5MB，边长 150-4096px
    """)