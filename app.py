# app.py — "🗨️그린(GREEN)톡톡💚" (B모드: 임베딩만, 생성 없음)
# - Google 스프레드시트 CSV 우선 로드(실패 시 로컬 엑셀 자동)
# - 컬럼명 자동 정규화(대/소문자, 공백/한글 별칭)
# - 카톡형 UI, 스몰톡, 첫 안내
# - ✅ images 컬럼 지원(+ 핫링크 차단 우회: 서버에서 이미지 바이트로 받아 표시, 프록시 백업)

import os, glob, re, time
import numpy as np
import pandas as pd
import streamlit as st

from google import genai
from google.genai import types

# ---- 외부 이미지 핫링크 우회용 의존성 ----
import requests
from io import BytesIO
from urllib.parse import urlparse, quote

# ===== (시연용) API Key 하드코딩 =====
API_KEY = "AIzaSyBklAdqxHazyHmEyJO6LD3kPzANiqc6u3o"

# ===== Google 스프레드시트 CSV URL (하드코딩) =====
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

# ===== 컬럼명 정규화 도우미 =====
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def _norm(s):
        s = str(s).strip().lower()
        s = re.sub(r"[\s\-\u00A0]+", "", s)  # 공백/하이픈/불가시 공백 제거
        return s
    df = df.rename(columns=_norm)
    # 한글/변형 별칭 매핑
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

# ===== KB 빌드 (✅ images 포함) =====
def build_kb(df: pd.DataFrame):
    df = _normalize_columns(df)

    rows = []
    for _, r in df.iterrows():
        answer = str(r.get("answer", "")).strip()
        if not answer:    # answer 필수
            continue
        intent = str(r.get("intent", "")).strip()
        images = _clean_images_field(r.get("images", ""))  # 선택 컬럼
        qs = [str(r.get(f"q{i}", "")).strip() for i in range(1,6) if str(r.get(f"q{i}", "")).strip()]
        if len(qs) < 3:   # 최소 3개 질문 변형
            continue
        for q in qs:
            rows.append({"intent": intent, "answer": answer, "q": q, "images": images})
    if not rows:
        return None
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

# ===== 엑셀 자동 인식 (로컬 폴백) =====
def auto_find_excel():
    cwd = os.getcwd()
    prefer = [os.path.join(cwd, "qa.xlsx"),
              os.path.join(cwd, "qa_template_min.xlsx"),
              os.path.join(cwd, "qa_template.xlsx")]
    for p in prefer:
        if os.path.isfile(p): return p
    others = glob.glob(os.path.join(cwd, "*.xlsx"))
    if others:
        others.sort(key=lambda p: os.path.getmtime(p), reverse=True)  # 최신 수정
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
            if kb:
                return kb, "gsheet"
            else:
                st.warning("스프레드시트에 유효한 Q&A가 없습니다. (answer 필수, q1~q3 이상 필요)")
        except Exception as e:
            st.warning(f"스프레드시트(CSV) 로드 실패, 로컬 엑셀을 시도합니다: {e}")

    path = auto_find_excel()
    if path:
        try:
            kb = load_excel(path)
            return kb, path
        except Exception as e:
            st.error(f"엑셀 로드 실패: {e}")
            return None, None
    return None, None

# ===== 스몰톡(규칙 기반) =====
def smalltalk_reply(text: str):
    t = text.strip().lower()
    if re.search(r"(너.*이름|이름이 뭐|who are you|what.*name)", t):
        return f"제 이름은 {BOT_NAME}이에요. 반가워요! 💚"
    if re.search(r"(안녕|안녕하세요|하이|헬로|hello|hi)", t):
        return f"안녕하세요! 저는 {BOT_NAME}이에요. 전 친구들이 설계한 질문에 대한 답변을 드리는 챗봇이에요. 많은 도움이 되었으면 좋겠어요 🙂"
    if re.search(r"(뭐(를)? 할 수|무엇을 할 수|무슨 기능|설명해줘|너.*할 수|역할|무얼 해)", t):
        return "엑셀(또는 시트)에 등록된 질문 변형과 가장 가까운 문장을 찾아, 등록된 ‘공식 답’을 알려주는 챗봇이에요. 등록이 없거나 유사도가 낮으면 답하지 않아요."
    if re.search(r"(몇살|나이|how old)", t):
        return "저는 나이는 없지만 언제나 수업을 도우려고 준비된 초등학교 챗봇이에요!"
    if re.search(r"(누가 만들|만든 사람|creator|developer)", t):
        return "저는 선생님과 함께 만들어진 GREEN 톡톡이에요. 교실에서 안전하게 쓰이도록 설계됐어요."
    if re.search(r"(무슨 말|이해.*안|모르겠)", t):
        return "조금만 더 구체적으로 말해줄래요? 예: ‘숙제 제출 시간 알려줘’처럼요."
    return None

# ===== 이미지 핫링크 차단 우회: 서버에서 바이트로 받아오기(+프록시 백업) =====
def fetch_image_bytes(url: str, timeout: int = 12):
    """
    1) 기본 요청
    2) UA만
    3) UA+Referer
    4) images.weserv.nl 프록시 백업
    성공 시 BytesIO, 실패 시 None
    """
    def _try(headers=None):
        try:
            resp = requests.get(url, headers=headers or {}, timeout=timeout)
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "").lower()
            if ("image" not in ctype and
                not resp.content.startswith(b"\x89PNG") and
                not resp.content.startswith(b"\xff\xd8")):  # JPEG/PNG 시그니처
                return None
            if len(resp.content) > 10 * 1024 * 1024:  # 10MB 방어
                return None
            return BytesIO(resp.content)
        except Exception:
            return None

    # 1) 기본
    buf = _try()
    if buf: return buf

    # 2) UA만
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
    buf = _try(ua)
    if buf: return buf

    # 3) UA + Referer + Accept
    parsed = urlparse(url)
    headers = dict(ua)
    headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}"
    headers["Accept"] = "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"
    buf = _try(headers)
    if buf: return buf

    # 4) 공개 프록시(weserv) 백업
    try:
        no_scheme = url.replace("https://", "").replace("http://", "")
        proxy_url = f"https://images.weserv.nl/?url={quote(no_scheme, safe='')}"
        resp = requests.get(proxy_url, timeout=timeout)
        resp.raise_for_status()
        if resp.content:
            return BytesIO(resp.content)
    except Exception:
        pass

    return None

# ===== 렌더 유틸 =====
def render_bot_message(text: str, images_field: str | None = None):
    st.markdown(f'<div class="msg-row left"><div class="msg bot">{text}</div></div>', unsafe_allow_html=True)
    if images_field:
        paths = [p.strip() for p in str(images_field).split(";") if p.strip()]
        if paths:
            n = min(len(paths), 3)
            cols = st.columns(n)
            for i in range(n):
                with cols[i]:
                    url = paths[i]
                    try:
                        if url.startswith("http"):
                            buf = fetch_image_bytes(url)
                            if buf:
                                st.image(buf, use_container_width=True)
                            else:
                                st.write("이미지를 불러올 수 없어요:")
                                st.markdown(f"<small><a href='{url}' target='_blank'>{url}</a></small>", unsafe_allow_html=True)
                        else:
                            st.image(url, use_container_width=True)
                    except Exception:
                        st.write("이미지를 불러올 수 없어요:")
                        st.markdown(f"<small><a href='{url}' target='_blank'>{url}</a></small>", unsafe_allow_html=True)

def render_user_message(text: str):
    st.markdown(f'<div class="msg-row right"><div class="msg user">{text}</div></div>', unsafe_allow_html=True)

# ===== 세션 상태 =====
if "kb" not in st.session_state:          st.session_state.kb = None
if "messages" not in st.session_state:    st.session_state.messages = []  # [{role,text,images?,ts}]
if "welcomed" not in st.session_state:    st.session_state.welcomed = False

# ===== KB 로드(스프레드시트 우선) + 첫 안내 =====
if st.session_state.kb is None:
    kb, source = load_excel_or_gsheet()
    if kb:
        st.session_state.kb = kb
        if source == "gsheet":
            st.caption("현재 지식베이스: Google 스프레드시트(CSV)에서 불러왔어요.")
        elif isinstance(source, str):
            st.caption(f"현재 지식베이스: 로컬 파일에서 불러왔어요. ({os.path.basename(source)})")
    else:
        st.info("GSHEET_CSV_URL에서 불러오지 못했습니다. 같은 폴더에 엑셀(.xlsx)을 두면 자동 인식합니다. (예: qa.xlsx)")

# 첫 안내 메시지
if st.session_state.kb and not st.session_state.welcomed:
    kb_rows, _ = st.session_state.kb
    intents = sorted(set([r["intent"] for r in kb_rows if r.get("intent")]))
    if intents:
        intents_txt = ", ".join([x.replace("_", " ") for x in intents if (x or "").strip()])
        first_msg = (
            f"안녕하세요! 저는 {BOT_NAME}이에요 💚\n\n"
            f"저는 이런 주제들에 대해 대답할 수 있어요:\n\n"
            f"👉 {intents_txt}\n\n무엇이 궁금하세요?"
        )
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

    reply = None
    reply_images = None

    # 1) KB 검색
    if st.session_state.kb:
        kb_rows, kb_mat = st.session_state.kb
        top_row, top_score = retrieve_top1(user_input, kb_rows, kb_mat)
        if top_row is not None and top_score >= SIM_THRESHOLD:
            reply = top_row["answer"]
            reply_images = _clean_images_field(top_row.get("images"))

    # 2) 스몰톡
    if reply is None:
        reply = smalltalk_reply(user_input)
        reply_images = None

    # 3) 최종 실패
    if reply is None:
        reply = NO_MATCH_MSG
        reply_images = None

    st.session_state.messages.append({
        "role":"assistant",
        "text":reply,
        "images":reply_images,
        "ts":time.time()
    })
    st.rerun()
