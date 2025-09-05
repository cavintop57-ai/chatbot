# app.py — "🗨️그린(GREEN)톡톡💚" (B모드: 임베딩만, 생성 없음)
# 요구사항 반영:
#  - 별도 파란 박스 제거, 페이지 채팅 영역 전체를 연파랑 배경으로
#  - 카톡형 말풍선(사용자 오른쪽, 봇 왼쪽)
#  - 스몰톡(인사/이름/기능/나이/만든 사람/이해못함)
#  - 엑셀 자동 인식(한글 파일명 포함, 우선순위 + 최신 수정)
#  - 첫 메시지: intent 목록을 친절히 안내 후 “무엇이 궁금하세요?”
#  - ✅ 엑셀 images 컬럼 지원(선택): "assets/a.png; https://.../b.jpg" 형식, 최대 3장 표시

import os, glob, re, time
import numpy as np
import pandas as pd
import streamlit as st

from google import genai
from google.genai import types

# ===== (시연용) API Key — 배포 시 환경변수/Secrets 사용 권장 =====
API_KEY = "AIzaSyBklAdqxHazyHmEyJO6LD3kPzANiqc6u3o"

# ===== 정책/문구 =====
SIM_THRESHOLD = 0.82
EMBED_MODEL   = "gemini-embedding-001"
EMBED_DIM     = 768
NO_MATCH_MSG  = "저는 친구들이 설계한 질문에 대한 답변만 드릴 수 있어요. 죄송해요. 친구들에게 질문을 추가해달라고 요청해보는 것은 어떨까요?"
APP_TITLE     = "🗨️그린(GREEN)톡톡💚"
APP_CAPTION   = "아산용연초등학교 6학년 4반 수업용 ChatBot입니다."
BOT_NAME      = "GREEN 톡톡"

st.set_page_config(page_title=APP_TITLE, page_icon="💚", layout="centered")

# ===== CSS (페이지 전역 채팅 영역을 연파랑 배경으로) =====
st.markdown("""
<style>
/* 중앙 폭 및 전체 배경 */
.main { max-width: 860px; margin: 0 auto; }
.block-container {
  background: #F6FAFF !important;   /* 연한 파랑 */
  border-radius: 12px;
  padding: 16px 24px 24px 24px;
}

/* 제목 영역 */
.title-wrap { padding: 8px 4px 0 4px; margin-bottom: 8px; background: transparent; }

/* 메시지 말풍선 */
.msg-row { display:flex; margin:8px 0; width:100%; }
.msg {
  display:inline-block; max-width:78%;
  padding:10px 14px; border-radius:16px;
  word-break:break-word; line-height:1.45;
  box-shadow:0 1px 0 rgba(0,0,0,0.03);
}
.right { justify-content:flex-end; }
.left  { justify-content:flex-start; }
.msg.user { background:#FEE500; color:#111; border-top-right-radius:6px; }
.msg.bot  { background:#fff;    color:#111; border-top-left-radius:6px; border:1px solid #ECF0F6; }

/* 봇 답변 아래 이미지 그리드 */
.media-row { display:flex; gap:8px; margin:6px 0 12px 0; }
.media-col { flex:1; }
.media-col img { border-radius:10px; border:1px solid #ECF0F6; }

/* chat_input 라벨 축소 */
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

# ===== KB 빌드 (✅ images 포함) =====
def build_kb(df: pd.DataFrame):
    rows = []
    for _, r in df.iterrows():
        answer = str(r.get("answer", "")).strip()
        if not answer:    # answer 필수
            continue
        intent = str(r.get("intent", "")).strip()
        images = _clean_images_field(r.get("images", ""))  # <<< 추가: images 컬럼(선택)
        qs = [str(r.get(f"q{i}", "")).strip() for i in range(1,6) if str(r.get(f"q{i}", "")).strip()]
        if len(qs) < 3:   # 최소 3개 질문 변형
            continue
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
        # 한글 파일명 포함, 최신 수정 파일 우선
        others.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return others[0]
    return None

def load_excel(path: str):
    df = pd.read_excel(path)
    return build_kb(df)

# ===== 스몰톡(규칙 기반) =====
def smalltalk_reply(text: str):
    t = text.strip().lower()

    # 이름/호칭
    if re.search(r"(너.*이름|이름이 뭐|who are you|what.*name)", t):
        return f"제 이름은 {BOT_NAME}이에요. 반가워요! 💚"

    # 인사
    if re.search(r"(안녕|안녕하세요|하이|헬로|hello|hi)", t):
        return f"안녕하세요! 저는 {BOT_NAME}이에요. 전 친구들이 설계한 질문에 대한 답변을 드리는 챗봇이에요. 많은 도움이 되었으면 좋겠어요 🙂"

    # 기능/역할
    if re.search(r"(뭐(를)? 할 수|무엇을 할 수|무슨 기능|설명해줘|너.*할 수|역할|무얼 해)", t):
        return "엑셀에 등록된 질문 변형과 가장 가까운 문장을 찾아, 등록된 ‘공식 답’을 그대로 알려주는 챗봇이에요. 등록이 없거나 유사도가 낮으면 답하지 않아요."

    # 나이
    if re.search(r"(몇살|나이|how old)", t):
        return "저는 나이는 없지만 언제나 수업을 도우려고 준비된 초등학교 챗봇이에요!"

    # 만든 사람
    if re.search(r"(누가 만들|만든 사람|creator|developer)", t):
        return "저는 선생님과 함께 만들어진 GREEN 톡톡이에요. 교실에서 안전하게 쓰이도록 설계됐어요."

    # 이해 못함
    if re.search(r"(무슨 말|이해.*안|모르겠)", t):
        return "조금만 더 구체적으로 말해줄래요? 예: ‘숙제 제출 시간 알려줘’처럼요."
    return None

# ===== 유틸: intent 목록을 자연어로 예쁘게 =====
def prettify_intents(intents: list[str]) -> str:
    # 예) "숙제_제출" -> "숙제 제출"
    cleaned = []
    for x in intents:
        x = (x or "").strip()
        if not x: continue
        x = x.replace("_", " ")
        cleaned.append(x)
    # 보기 좋게 쉼표로 연결
    return ", ".join(cleaned)

# ===== (새로 추가) 렌더 유틸: 봇 메시지 + 이미지 =====
def render_bot_message(text: str, images_field: str | None = None):
    # 텍스트 버블
    st.markdown(f'<div class="msg-row left"><div class="msg bot">{text}</div></div>', unsafe_allow_html=True)
    # 이미지가 있으면 아래에 그리드로 표시 (최대 3장)
    if images_field:
        paths = [p.strip() for p in str(images_field).split(";") if p.strip()]
        if paths:
            n = min(len(paths), 3)
            cols = st.columns(n)
            for i in range(n):
                with cols[i]:
                    try:
                        st.image(paths[i], use_container_width=True)
                    except Exception:
                        st.write("이미지를 불러올 수 없어요:", paths[i])

def render_user_message(text: str):
    st.markdown(f'<div class="msg-row right"><div class="msg user">{text}</div></div>', unsafe_allow_html=True)

# ===== 세션 상태 =====
if "kb" not in st.session_state:          st.session_state.kb = None
if "messages" not in st.session_state:    st.session_state.messages = []  # [{role,text,images?,ts}]
if "welcomed" not in st.session_state:    st.session_state.welcomed = False  # 첫 안내 메시지 중복 방지

# ===== KB 자동 로드 + 첫 안내 =====
if st.session_state.kb is None:
    xls = auto_find_excel()
    if xls:
        try:
            kb = load_excel(xls)
            st.session_state.kb = kb
        except Exception as e:
            st.error(f"엑셀 자동 로드 실패: {e}")
    else:
        st.info("같은 폴더에 엑셀(.xlsx)을 두면 자동 인식합니다. (예: qa.xlsx)")

# KB가 있고 아직 환영 메시지를 안 보냈다면 intents로 첫 메시지 안내
if st.session_state.kb and not st.session_state.welcomed:
    kb_rows, _ = st.session_state.kb
    intents = sorted(set([r["intent"] for r in kb_rows if r.get("intent")]))
    if intents:
        intents_txt = prettify_intents(intents)
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
        # Assistant 메시지는 images 가능
        render_bot_message(m["text"], m.get("images"))

# ===== 입력창 =====
user_input = st.chat_input("질문을 입력하세요… (예: 숙제 언제까지 내나요?)", key="chat_input")
if user_input:
    # 사용자 메시지 저장
    st.session_state.messages.append({"role":"user","text":user_input,"ts":time.time()})

    # 1) KB 검색
    reply = None
    reply_images = None
    if st.session_state.kb:
        kb_rows, kb_mat = st.session_state.kb
        top_row, top_score = retrieve_top1(user_input, kb_rows, kb_mat)
        if top_row is not None and top_score >= SIM_THRESHOLD:
            reply = top_row["answer"]
            reply_images = _clean_images_field(top_row.get("images"))  # <<< 매칭된 행의 images 사용

    # 2) 스몰톡
    if reply is None:
        reply = smalltalk_reply(user_input)
        reply_images = None  # 스몰톡에는 이미지 미첨부

    # 3) 최종 실패
    if reply is None:
        reply = NO_MATCH_MSG
        reply_images = None

    # 저장(이미지 포함) & 즉시 갱신
    st.session_state.messages.append({
        "role":"assistant",
        "text":reply,
        "images":reply_images,
        "ts":time.time()
    })
    st.rerun()
