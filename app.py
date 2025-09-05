# app.py — "🗨️그린(GREEN)톡톡💚"
# - Google 스프레드시트 CSV 우선 로드(실패 시 로컬 엑셀 자동)
# - 컬럼명 자동 정규화
# - 카톡형 UI, 스몰톡, 첫 안내
# - ✅ images: 서버에서 bytes로 받아 표시(UA/Referer 재시도) → 프록시(weserv/duck) 백업
#             썸네일(말풍선 아래) + "크게 보기" 모달, 최후엔 새 탭 버튼 폴백
#             (st.image에 BytesIO 대신 raw bytes 전달로 안정화)

import os, glob, re, time
import numpy as np
import pandas as pd
import streamlit as st

from google import genai
from google.genai import types

import requests
from urllib.parse import urlparse, quote

# ===== API Key (시연용 하드코딩) =====
API_KEY = "AIzaSyBklAdqxHazyHmEyJO6LD3kPzANiqc6u3o"

# ===== Google 스프레드시트 CSV URL =====
GSHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRuh7Qmg1YFrj1IleUs0QJmCBFfb7Cgu_7prj-LmrcovxD-s2ON1Q86ENi27BUwZdpqOqrCCdJDrKmW/pub?output=csv"

# ===== 정책/문구 =====
SIM_THRESHOLD = 0.82
EMBED_MODEL   = "gemini-embedding-001"
EMBED_DIM     = 768
NO_MATCH_MSG  = "저는 친구들이 설계한 질문에 대한 답변만 드릴 수 있어요. 죄송해요. 친구들에게 질문을 추가해달라고 요청해보는 것은 어떨까요?"
APP_TITLE     = "🗨️그린(GREEN)톡톡💚"
APP_CAPTION   = "아산용연초등학교 6학년 4반 수업용 ChatBot입니다."
BOT_NAME      = "GREEN 톡톡"

st.set_page_config(page_title=APP_TITLE, page_icon="💚", layout="centered")

# ===== CSS =====
st.markdown("""
<style>
.main { max-width: 860px; margin: 0 auto; }
.block-container { background: #F6FAFF !important; border-radius: 12px; padding: 16px 24px 24px; }
.title-wrap { padding: 8px 4px 0 4px; margin-bottom: 8px; background: transparent; }

.msg-row { display:flex; margin:8px 0; width:100%; }
.msg { display:inline-block; max-width:78%; padding:10px 14px; border-radius:16px;
       word-break:break-word; line-height:1.45; box-shadow:0 1px 0 rgba(0,0,0,0.03); }
.right { justify-content:flex-end; } .left { justify-content:flex-start; }
.msg.user { background:#FEE500; color:#111; border-top-right-radius:6px; }
.msg.bot  { background:#fff; color:#111; border-top-left-radius:6px; border:1px solid #ECF0F6; }

.small-note { font-size: 0.9rem; color: #4a5568; margin: 6px 0 4px 4px; }
.thumb-wrap { margin: 6px 0 2px 6px; }
.thumb-btn { margin-top: 4px; }

label[for="chat_input"] { font-size:0; }
</style>
""", unsafe_allow_html=True)

# ===== 제목 =====
st.markdown('<div class="title-wrap">', unsafe_allow_html=True)
st.title(APP_TITLE)
st.caption(APP_CAPTION)
st.markdown('</div>', unsafe_allow_html=True)

# ===== Gemini client =====
try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    st.error(f"Gemini 클라이언트 초기화 실패: {e}")
    st.stop()

# ===== 임베딩 유틸 =====
def embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", out_dim=EMBED_DIM):
    if not texts: return []
    res = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=out_dim),
    )
    return [np.array(e.values, dtype=np.float32) for e in res.embeddings]

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norm

# ===== 도우미: images 필드 정리 =====
def _clean_images_field(val) -> str:
    if val is None: return ""
    s = str(val).strip()
    if not s or s.lower() == "nan": return ""
    return s

# ===== 컬럼명 정규화 =====
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def _norm(s):
        s = str(s).strip().lower()
        s = re.sub(r"[\s\-\u00A0]+", "", s)
        return s
    df = df.rename(columns=_norm)
    alias = {
        "의도": "intent", "질문그룹": "intent",
        "답": "answer", "답변": "answer",
        "이미지": "images", "사진": "images",
        "질문1": "q1", "질문2": "q2", "질문3": "q3", "질문4": "q4", "질문5": "q5",
        "q01": "q1", "q02": "q2", "q03": "q3", "q04": "q4", "q05": "q5",
        "q1": "q1", "q2": "q2", "q3": "q3", "q4": "q4", "q5": "q5",
        "intent": "intent", "answer": "answer", "images": "images"
    }
    for k, v in alias.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    return df

# ===== KB 빌드 =====
def build_kb(df: pd.DataFrame):
    df = _normalize_columns(df)
    rows = []
    for _, r in df.iterrows():
        answer = str(r.get("answer", "")).strip()
        if not answer: continue
        intent = str(r.get("intent", "")).strip()
        images = _clean_images_field(r.get("images", ""))
        qs = [str(r.get(f"q{i}", "")).strip() for i in range(1,6) if str(r.get(f"q{i}", "")).strip()]
        if len(qs) < 3: continue
        for q in qs:
            rows.append({"intent": intent, "answer": answer, "q": q, "images": images})
    if not rows: return None
    vecs = embed_texts([r["q"] for r in rows], task_type="RETRIEVAL_DOCUMENT")
    mat  = l2_normalize(np.vstack(vecs))
    return rows, mat

def retrieve_top1(query: str, kb_rows, kb_mat):
    if not kb_rows: return None, 0.0
    qvec = embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
    qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
    scores = kb_mat @ qvec
    idx = int(np.argmax(scores))
    return kb_rows[idx], float(scores[idx])

# ===== 엑셀 자동 인식 =====
def auto_find_excel():
    cwd = os.getcwd()
    prefer = [os.path.join(cwd, "qa.xlsx"),
              os.path.join(cwd, "qa_template_min.xlsx"),
              os.path.join(cwd, "qa_template.xlsx")]
    for p in prefer:
        if os.path.isfile(p): return p
    others = glob.glob(os.path.join(cwd, "*.xlsx"))
    if others:
        others.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return others[0]
    return None

def load_excel(path: str):
    df = pd.read_excel(path)
    return build_kb(df)

# ===== 스프레드시트 우선 로더 =====
def load_excel_or_gsheet():
    if GSHEET_CSV_URL:
        try:
            df = pd.read_csv(GSHEET_CSV_URL, encoding="utf-8", keep_default_na=False)
            kb = build_kb(df)
            if kb: return kb, "gsheet"
            else: st.warning("스프레드시트에 유효한 Q&A가 없습니다.")
        except Exception as e:
            st.warning(f"스프레드시트 로드 실패: {e}")
    path = auto_find_excel()
    if path:
        try:
            kb = load_excel(path)
            return kb, path
        except Exception as e:
            st.error(f"엑셀 로드 실패: {e}")
    return None, None

# ===== 이미지 fetch/프록시 =====
def proxy_urls(url: str):
    no_scheme = url.replace("https://", "").replace("http://", "")
    return [
        f"https://images.weserv.nl/?url={quote(no_scheme, safe='')}&w=1200&output=jpg",
        f"https://proxy.duckduckgo.com/iu/?u={quote(url, safe='')}&f=1",
    ]

def fetch_image_bytes(url: str, timeout: int = 15, max_bytes: int = 20 * 1024 * 1024):
    """직접 요청 → UA → UA+Referer → 프록시 2종 순으로 바이트 획득.
       성공 시 (bytes, mime) 반환, 실패 시 (None, None)"""
    def _try(u, headers=None):
        try:
            resp = requests.get(u, headers=headers or {}, timeout=timeout)
            resp.raise_for_status()
            data = resp.content or b""
            if not data or len(data) > max_bytes:
                return None, None
            ctype = (resp.headers.get("Content-Type") or "").lower()
            # 이미지 시그니처/타입 대충 확인
            sig_ok = (
                data.startswith(b"\x89PNG") or     # PNG
                data.startswith(b"\xff\xd8") or     # JPEG
                data[:12].lower().startswith(b"riff") or  # WEBP
                "image" in ctype
            )
            if not sig_ok:
                return None, None
            return data, ctype
        except Exception:
            return None, None

    # 1) 기본
    data, mime = _try(url)
    if data: return data, mime
    # 2) UA
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
    data, mime = _try(url, ua)
    if data: return data, mime
    # 3) UA + Referer + Accept
    parsed = urlparse(url)
    hdr = dict(ua)
    hdr["Referer"] = f"{parsed.scheme}://{parsed.netloc}"
    hdr["Accept"]  = "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"
    data, mime = _try(url, hdr)
    if data: return data, mime
    # 4) 프록시들
    for p in proxy_urls(url):
        data, mime = _try(p, ua)
        if data: return data, mime
    return None, None

# ===== 모달(데코레이터) — 크게 보기 =====
@st.dialog("이미지 미리보기", width="large")
def image_modal(url: str):
    data, _ = fetch_image_bytes(url, timeout=18, max_bytes=25 * 1024 * 1024)
    if data:
        st.image(data, use_container_width=True)
    else:
        st.markdown(
            "<div class='small-note'>이 이미지는 보안 정책 때문에 직접 표시가 어려워요. "
            "아래 버튼을 눌러 <b>새 탭</b>에서 확인해 보세요 👇</div>",
            unsafe_allow_html=True
        )
        st.link_button("이미지 열기 (새 탭)", url)

# ===== 렌더 유틸 =====
def render_bot_message(text: str, images_field: str | None = None):
    # 텍스트 말풍선
    st.markdown(f'<div class="msg-row left"><div class="msg bot">{text}</div></div>', unsafe_allow_html=True)

    # 이미지 썸네일 (최대 3장)
    if images_field:
        paths = [p.strip() for p in str(images_field).split(";") if p.strip()]
        if paths:
            cols = st.columns(min(len(paths), 3))
            for i, url in enumerate(paths[:3]):
                with cols[i % len(cols)]:
                    data, _ = fetch_image_bytes(url)
                    if data:
                        st.markdown("<div class='thumb-wrap'>", unsafe_allow_html=True)
                        try:
                            st.image(data, use_container_width=True)
                            if st.button("크게 보기", key=f"zoom_{hash(url)}_{i}", help="모달로 크게 보기"):
                                image_modal(url)
                        except Exception:
                            # st.image에서 또 예외가 나면 버튼 폴백
                            st.markdown(
                                "<div class='small-note'>이미지를 직접 표시하지 못했어요. "
                                "아래 버튼을 눌러 <b>새 탭</b>에서 확인해 보세요 👇</div>",
                                unsafe_allow_html=True
                            )
                            st.link_button("이미지 열기 (새 탭)", url)
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        # 인라인 실패 → 안내 + 새 탭 버튼
                        st.markdown(
                            "<div class='small-note'>일부 사이트는 보안 정책으로 이미지 임베드를 막아요. "
                            "아래 버튼을 눌러 <b>새 탭</b>에서 확인해 보세요 👇</div>",
                            unsafe_allow_html=True
                        )
                        st.link_button("이미지 열기 (새 탭)", url)

def render_user_message(text: str):
    st.markdown(f'<div class="msg-row right"><div class="msg user">{text}</div></div>', unsafe_allow_html=True)

# ===== 세션 상태 =====
if "kb" not in st.session_state:          st.session_state.kb = None
if "messages" not in st.session_state:    st.session_state.messages = []
if "welcomed" not in st.session_state:    st.session_state.welcomed = False

# ===== KB 로드 =====
if st.session_state.kb is None:
    kb, source = load_excel_or_gsheet()
    if kb: st.session_state.kb = kb

# 첫 안내 메시지
if st.session_state.kb and not st.session_state.welcomed:
    kb_rows, _ = st.session_state.kb
    intents = sorted(set([r["intent"] for r in kb_rows if r.get("intent")]))
    if intents:
        intents_txt = ", ".join([x.replace("_", " ") for x in intents if (x or "").strip()])
        first_msg = f"안녕하세요! 저는 {BOT_NAME}이에요 💚\n\n제가 대답할 수 있는 주제는 👉 {intents_txt}\n\n무엇이 궁금하세요?"
    else:
        first_msg = f"안녕하세요! 저는 {BOT_NAME}이에요 💚 무엇이 궁금하세요?"
    st.session_state.messages.append({"role": "assistant", "text": first_msg, "ts": time.time()})
    st.session_state.welcomed = True

# ===== 과거 메시지 렌더 =====
for m in st.session_state.messages:
    if m["role"] == "user":
        render_user_message(m["text"])
    else:
        render_bot_message(m["text"], m.get("images"))

# ===== 입력창 =====
user_input = st.chat_input("질문을 입력하세요… (예: 숙제 언제까지 내나요?)", key="chat_input")
if user_input:
    st.session_state.messages.append({"role":"user","text":user_input,"ts":time.time()})
    reply, reply_images = None, None

    if st.session_state.kb:
        kb_rows, kb_mat = st.session_state.kb
        top_row, top_score = retrieve_top1(user_input, kb_rows, kb_mat)
        if top_row and top_score >= SIM_THRESHOLD:
            reply = top_row["answer"]
            reply_images = _clean_images_field(top_row.get("images"))

    if reply is None:
        reply = smalltalk_reply(user_input)

    if reply is None:
        reply = NO_MATCH_MSG

    st.session_state.messages.append({"role":"assistant","text":reply,"images":reply_images,"ts":time.time()})
    st.rerun()
