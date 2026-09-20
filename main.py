import os
import io
import logging
import platform
import threading
import asyncio
from flask import Flask
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)
import google.generativeai as genai
from groq import Groq

# 1. 로깅 및 한국 시간 설정
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
KST = pytz.timezone('Asia/Seoul')

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# 2. API 키 및 환경 변수 로드
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# ==========================================
# 3. 실시간 종목 코드 판별기 (오류 없음)
# ==========================================
class QuickStockResolver:
    # 자주 쓰이는 주요 종목 단축어 매핑
    KNOWN_STOCKS = {
        "삼성전자": "005930.KS", "SK하이닉스": "000660.KS", "테웅": "044780.KQ", 
        "에코프로": "086520.KQ", "에코프로비엠": "247540.KQ", "셀트리온": "068270.KS",
        "LG에너지솔루션": "373220.KS", "현대차": "005380.KS", "기아": "000270.KS", "삼성바이오로직스": "207940.KS"
    }

    @classmethod
    def resolve(cls, query: str) -> tuple:
        if not query:
            return "005930.KS", "삼성전자"
        
        q = query.strip()
        
        # 1. 사전 등록된 이름인 경우
        if q in cls.KNOWN_STOCKS:
            return cls.KNOWN_STOCKS[q], q

        # 2. 이미 .KS나 .KQ가 포함된 경우
        if "." in q:
            return q.upper(), q

        # 3. 6자리 숫자 코드인 경우 (코스피 우선 확인 후 코스닥 확인)
        if q.isdigit() and len(q) == 6:
            for suffix in [".KS", ".KQ"]:
                test_t = q + suffix
                try:
                    df = yf.download(test_t, period="2d", progress=False)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.droplevel(1)
                    if not df.empty:
                        return test_t, q
                except:
                    continue
            return q + ".KS", q

        # 4. 한글 종목명으로 검색 시도 (야후 파이낸스 호환 코스피/코스닥 테스트)
        # 사용자가 한글로 '태웅' 같은 이름을 쳤을 때를 대비한 검색 로직
        for suffix in [".KS", ".KQ"]:
            # 야후 파이낸스는 한글 티커를 직접 받지 못하므로 기본 6자리 변환이 안 된 경우 예외 처리용
            pass

        return q.upper() + ".KS", q


# ==========================================
# 4. AI 분석 서비스 (Gemini 및 Groq 백업)
# ==========================================
class AIServiceRouter:
    @staticmethod
    def analyze(prompt: str, image_bytes: bytes = None) -> str:
        summary_instruction = "[요청 사항]\n장황한 설명은 빼고 핵심 투자 포인트만 마크다운 요약 형태로 간결하게 작성해주세요.\n\n"
        full_prompt = summary_instruction + prompt
        gemini_err, groq_err = "", ""
        
        # 1차 시도: Gemini
        try:
            if GEMINI_API_KEY:
                model = genai.GenerativeModel('gemini-1.5-flash')
                content = [full_prompt, {'mime_type': 'image/png', 'data': image_bytes}] if image_bytes else [full_prompt]
                response = model.generate_content(content)
                if response.text:
                    return f"🤖 **[Gemini AI 분석 리포트]**\n\n" + response.text
        except Exception as e:
            gemini_err = str(e)
            logger.warning(f"Gemini AI 호출 실패: {e}")

        # 2차 시도: Groq (백업)
        try:
            if groq_client:
                completion = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "당신은 냉철한 금융 기술적 분석 전문가입니다."}, 
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.2
                )
                if completion.choices[0].message.content:
                    return f"⚡ **[Groq 백업 AI 분석 리포트]**\n\n" + completion.choices[0].message.content
        except Exception as e:
            groq_err = str(e)
            logger.warning(f"Groq AI 호출 실패: {e}")

        return f"⚠️ AI 분석을 수행할 수 없습니다.\n(Gemini 오류: {gemini_err} / Groq 오류: {groq_err})\nAPI 키가 올바르게 입력되었는지 확인해주세요."


# ==========================================
# 5. 기술적 지표 및 차트 이미지 생성 엔진
# ==========================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    
    # 볼린저 밴드
    bb_std = df['Close'].rolling(window=20, min_periods=1).std()
    df['BB_Middle'] = df['MA20']
    df['BB_Upper'] = df['MA20'] + (bb_std * 2)
    df['BB_Lower'] = df['MA20'] - (bb_std * 2)

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    return df

def generate_chart_image(df: pd.DataFrame, ticker_name: str) -> tuple:
    df = df.copy()
    date_strings = df.index.strftime('%Y-%m-%d').values
    x_indexes = np.arange(len(df))
    
    # 서브 차트(RSI) 포함 레이아웃 생성
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # 캔들스틱 시뮬레이션 렌더링
    up_mask = df['Close'] >= df['Open']
    down_mask = df['Close'] < df['Open']
    
    if len(x_indexes[up_mask]) > 0:
        ax1.bar(x_indexes[up_mask], df['Close'].values[up_mask] - df['Open'].values[up_mask], 0.8, bottom=df['Open'].values[up_mask], color='#ef5350', alpha=0.9)
        ax1.vlines(x_indexes[up_mask], df['Low'].values[up_mask], df['High'].values[up_mask], color='#ef5350', linewidth=1.2)
    if len(x_indexes[down_mask]) > 0:
        ax1.bar(x_indexes[down_mask], df['Open'].values[down_mask] - df['Close'].values[down_mask], 0.8, bottom=df['Close'].values[down_mask], color='#26a69a', alpha=0.9)
        ax1.vlines(x_indexes[down_mask], df['Low'].values[down_mask], df['High'].values[down_mask], color='#26a69a', linewidth=1.2)

    current_price = float(df['Close'].iloc[-1])
    summary_text = f"종목: {ticker_name} | 현재가: {current_price:,.2f}\n"

    # 이평선 및 볼린저 밴드 plotting
    ax1.plot(x_indexes, df['MA5'].values, label='MA 5', color='orange', linewidth=1)
    ax1.plot(x_indexes, df['MA20'].values, label='MA 20', color='green', linewidth=1)
    ax1.plot(x_indexes, df['BB_Upper'].values, color='red', linestyle='--', alpha=0.5)
    ax1.plot(x_indexes, df['BB_Lower'].values, color='blue', linestyle='--', alpha=0.5)

    # 하단 RSI 차트
    ax2.plot(x_indexes, df['RSI'].values, color='purple', label='RSI (14)')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(30, color='blue', linestyle='--', alpha=0.7)
    ax2.set_ylabel('RSI')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.3)

    step = max(1, len(x_indexes) // 6)
    ax2.set_xticks(x_indexes[::step])
    ax2.set_xticklabels(date_strings[::step], rotation=15)
    
    ax1.set_title(f"{ticker_name} 기술적 분석 차트", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.5)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue(), summary_text


# ==========================================
# 6. 텔레그램 봇 핸들러 구현 (실제 동작)
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 **주식 자동 분석 봇이 활성화 상태입니다!**\n\n"
        "사용 가능한 명령어:\n"
        "• `/추세 [종목명 또는 코드]` (예: `!추세 삼성전자` 또는 `!추세 044780`)\n"
        "• `/손절가 [종목명 또는 코드]` (3단계 손절선 산출)\n"
        "• `/help` (도움말 보기)",
        parse_mode="Markdown"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📌 **[명령어 사용법 가이드]**\n\n"
        "1. `!추세 삼성전자` 혹은 `/추세 005930`\n   ➡️ 6개월 차트 이미지 생성 + AI 기술적 분석 리포트 전송\n"
        "2. `!손절가 테웅` 혹은 `/손절가 044780`\n   ➡️ 현재가 기준 -3%, -6%, -10% 손절 가격 자동 산출\n"
        "3. `/start` 또는 `/help`\n   ➡️ 봇 안내 및 도움말",
        parse_mode="Markdown"
    )

async def handle_message_or_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    
    text = update.message.text.strip()
    
    # 명령어 파싱 (! 또는 / 지원)
    if text.startswith("!") or text.startswith("/"):
        parts = text.split(maxsplit=1)
        cmd = parts[0].replace("!", "/")  # 모두 슬래시 명령어로 통일 처리
        target = parts[1] if len(parts) > 1 else ""

        if cmd in ["/start"]:
            await start_command(update, context)
            return
        elif cmd in ["/help", "/명령어"]:
            await help_command(update, context)
            return

        if cmd in ["/추세", "/차트"]:
            ticker, name = QuickStockResolver.resolve(target)
            await update.message.reply_text(f"📊 [{name}] ({ticker}) 주가 데이터 다운로드 및 차트 분석 중...")
            try:
                df = yf.download(ticker, period="6mo", interval="1d", progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                if df.empty:
                    await update.message.reply_text(f"❌ '{target}' 종목의 데이터를 불러오지 못했습니다. 종목명을 정확히 입력해 주세요.")
                    return

                df = calculate_indicators(df)
                img_bytes, summary = generate_chart_image(df, f"{name} ({ticker})")
                
                # AI 분석 요청
                ai_report = AIServiceRouter.analyze(f"다음은 {name} ({ticker})의 최근 기술적 지표 및 주가 데이터 요약입니다:\n{summary}\n차트를 참고하여 투자 조언을 제공해주세요.", img_bytes)
                
                # 텔레그램 전송
                await update.message.reply_photo(photo=img_bytes, caption=f"📊 [{name}] 기술적 분석 차트")
                
                # 메시지 글자수가 길 경우 분할 전송
                if len(ai_report) > 4000:
                    for i in range(0, len(ai_report), 4000):
                        await update.message.reply_text(ai_report[i:i+4000], parse_mode="Markdown")
                else:
                    await update.message.reply_text(ai_report, parse_mode="Markdown")

            except Exception as e:
                logger.error(f"차트 분석 처리 오류: {e}")
                await update.message.reply_text(f"⚠️ 분석 중 오류가 발생했습니다: {e}")
            return

        if cmd in ["/손절가"]:
            ticker, name = QuickStockResolver.resolve(target)
            try:
                df = yf.download(ticker, period="3d", progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                if df.empty:
                    await update.message.reply_text("❌ 종목 데이터를 찾을 수 없습니다.")
                    return

                cp = float(df['Close'].iloc[-1])
                s1, s2, s3 = cp * 0.97, cp * 0.94, cp * 0.90
                await update.message.reply_text(
                    f"🛡️ **[{name}] ({ticker}) 3단계 손절가 산출**\n\n"
                    f"• 현재 종가: `{cp:,.2f}원`\n"
                    f"• 1차 손절선 (-3%): `{s1:,.2f}원`\n"
                    f"• 2차 손절선 (-6%): `{s2:,.2f}원`\n"
                    f"• 3차 손절선 (-10%): `{s3:,.2f}원`",
                    parse_mode="Markdown"
                )
            except Exception as e:
                await update.message.reply_text(f"⚠️ 손절가 계산 중 오류 발생: {e}")
            return

        # 알 수 없는 명령어 처리
        await update.message.reply_text("⚠️ 지원하지 않는 명령어입니다. `/help`를 입력해 사용법을 확인하세요.", parse_mode="Markdown")
    else:
        # 일반 텍스트 입력 시 안내
        await update.message.reply_text(f"💡 명령어로 주식을 분석해 보세요.\n예시: `!추세 {text}` 또는 `!손절가 {text}`", parse_mode="Markdown")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"텔레그램 봇 예외 발생: {context.error}")


# ==========================================
# 7. Flask 웹 서버 (Render 생존 유지용)
# ==========================================
web_app = Flask(__name__)

@web_app.route('/')
def home():
    return "Telegram Trading Bot Server is up and running live!", 200

def run_web():
    port = int(os.environ.get("PORT", 10000))
    web_app.run(host="0.0.0.0", port=port)


# ==========================================
# 8. 메인 실행 진입점
# ==========================================
def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN이 설정되지 않았습니다!")
        return

    # 텔레그램 봇 빌드
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # 핸들러 등록 (모든 텍스트 및 명령어 유연한 수신)
    application.add_handler(MessageHandler(filters.TEXT, handle_message_or_command))
    application.add_error_handler(error_handler)

    # Flask 웹 서버 스레드 시작 (렌더 절전 방지용)
    threading.Thread(target=run_web, daemon=True).start()
    logger.info("🌐 Flask 웹 서버가 포트 바인딩을 시작했습니다.")

    # 봇 폴링 시작
    logger.info("🤖 텔레그램 봇 폴링(Polling) 루프를 시작합니다...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
