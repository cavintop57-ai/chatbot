# app.py — "🗨️그린(GREEN)톡톡💚"
# - Google 스프레드시트 CSV 우선 로드(실패 시 로컬 엑셀 자동)
# - 컬럼명 자동 정규화
# - 카톡형 UI, 스몰톡, 첫 안내
# - ✅ images 컬럼: 직접 표시 시도 → 실패하면 안내 문구 + "이미지 열기 (새 탭)" 버튼 제공

import os, glob, re, time
import numpy as np
import pandas as pd
import streamlit as st

from google import genai
from google.genai import types

# (이미지 직접 표시 실패 시 버튼만 제공하므로 외부 프록시는 사용하지 않습니다)
# 필요시 requests 등 추가 의존성 없이 동작

# ===== API Key =====
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

# ===== 스몰톡 =====
def smalltalk_reply(text: str):
    t = text.strip().lower()
    if re.search(r"(너.*이름|이름이 뭐|who are you|what.*name)", t):
        return f"제 이름은 {BOT_NAME}이에요. 반가워요! 💚"
    if re.search(r"(안녕|안녕하세요|하이|헬로|hello|hi)", t):
        return f"안녕하세요! 저는 {BOT_NAME}이에요. 전 친구들이 설계한 질문에 대한 답변을 드리는 챗봇이에요 🙂"
    if re.search(r"(뭐(를)? 할 수|무엇을 할 수|무슨 기능|설명해줘|역할)", t):
        return "저는 시트에 등록된 질문과 답변을 연결해주는 챗봇이에요."
    if re.search(r"(몇살|나이|how old)", t):
        return "저는 나이는 없지만 언제나 수업을 도우려고 준비된 챗봇이에요!"
    if re.search(r"(누가 만들|만든 사람|creator|developer)", t):
        return "저는 선생님과 함께 만들어진 GREEN 톡톡이에요."
    return None

# ===== 이미지 표시: 실패 시 버튼으로 유도 =====
def render_bot_message(text: str, images_field: str | None = None):
    st.markdown(f'<div class="msg-row left"><div class="msg bot">{text}</div></div>', unsafe_allow_html=True)
    if images_field:
        paths = [p.strip() for p in str(images_field).split(";") if p.strip()]
        if paths:
            cols = st.columns(min(len(paths), 3))
            for i, url in enumerate(paths[:3]):
                with cols[i % len(cols)]:
                    try:
                        st.image(url, use_container_width=True)
                    except Exception:
                        st.markdown("<div class='small-note'>사진을 준비했어요. 아래 버튼을 누르면 답변에 준비된 사진을 확인하실 수 있습니다~! 👇</div>", unsafe_allow_html=True)
                        # Streamlit 1.25+ : 링크 버튼
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
