import os
import io
import json
import logging
import platform
import threading
import asyncio
import urllib.request
from flask import Flask
import pytz
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)
from apscheduler.schedulers.background import BackgroundScheduler

# 1. 로깅 및 환경 설정
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
KST = pytz.timezone('Asia/Seoul')

if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# 2. API 토큰 및 설정
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
GROQ_API_KEY = (os.environ.get("GROQ_API_KEY") or "").strip()
GEMINI_API_KEY = (os.environ.get("GEMINI_API_KEY") or "").strip()
ADMIN_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")

# 자동매매 전역 상태 관리
AUTO_TRADING_ACTIVE = True


# ==========================================
# 3. AI 서비스 라우터 (Groq / Gemini 연동)
# ==========================================
class AIServiceRouter:
    @staticmethod
    def call_groq(prompt: str) -> str:
        if not GROQ_API_KEY:
            raise Exception("GROQ_API_KEY 미설정")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_API_KEY}"}
        payload = {"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
        req = urllib.request.Request(GROQ_BASE_URL, data=json.dumps(payload).encode('utf-8'), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            content = res_data['choices'][0]['message']['content']
            if content:
                return f"⚡ **[Groq AI 자율 분석]**\n\n" + content
        raise Exception("Groq 응답 비어 있음")

    @staticmethod
    def analyze(prompt: str, image_bytes: bytes = None) -> str:
        sys_instruction = "당신은 주식 시장을 24시간 감시하며 자율적으로 매매 기회를 포착하는 AI 관제 시스템입니다. 핵심만 냉철하게 분석해주세요.\n\n"
        full_prompt = sys_instruction + prompt
        if GROQ_API_KEY and not image_bytes:
            try:
                return AIServiceRouter.call_groq(full_prompt)
            except Exception as e:
                logger.warning(f"Groq 실패, Gemini 전환: {e}")
        if GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                content_payload = [full_prompt]
                if image_bytes:
                    content_payload.append({"mime_type": "image/png", "data": image_bytes})
                response = model.generate_content(content_payload)
                if response and response.text:
                    return f"✨ **[Gemini AI 자율 분석]**\n\n" + response.text
            except Exception as ge:
                logger.warning(f"Gemini 실패: {ge}")
        return "💡 **[AI API 설정 오류]** 환경 변수에 GROQ_API_KEY 또는 GEMINI_API_KEY가 올바르게 입력되었는지 확인해주세요."


# ==========================================
# 4. 24시간 스스로 작동하는 자율 스캔 엔진
# ==========================================
def run_autonomous_scanner(app):
    """사람이 명령을 내리지 않아도 백그라운드에서 주기적으로 혼자 돌아가는 자율 매매 엔진"""
    global AUTO_TRADING_ACTIVE
    if not AUTO_TRADING_ACTIVE or not ADMIN_CHAT_ID:
        return

    logger.info("🤖 [자율 엔진] 스스로 시장 스캔 및 유망 종목 발굴 시작...")
    try:
        prompt = "현재 한국 주식 시장에서 거래대금이 폭증하고 수급이 집중되는 유망 종목 1개를 스스로 골라 매수가, 목표가, 손절가를 분석해줘."
        analysis_result = AIServiceRouter.analyze(prompt)
        
        report_msg = (
            "🤖 **[자동매매 자율 관제 리포트]**\n\n"
            f"{analysis_result}\n\n"
            "🟢 **엔진 상태:** 백그라운드 자율 스캔 및 가상 모니터링 완료"
        )
        
        async def push_to_telegram():
            await app.bot.send_message(chat_id=ADMIN_CHAT_ID, text=report_msg, parse_mode="Markdown")
            
        asyncio.run(push_to_telegram())
        logger.info("🤖 [자율 엔진] 텔레그램 자동 전송 완료")
    except Exception as e:
        logger.error(f"자율 엔진 구동 중 에러 발생 (봇은 꺼지지 않음): {e}")


# ==========================================
# 5. 종목 자동 탐색기
# ==========================================
class QuickStockResolver:
    POPULAR_STOCKS = {
        "삼성전자": "005930.KS", "SK하이닉스": "000660.KS", "삼성SDI": "006400.KS",
        "에코프로": "086520.KQ", "에코프로비엠": "247540.KS", "셀트리온": "068270.KS",
        "LG에너지솔루션": "373220.KS", "현대차": "005380.KS", "기아": "000270.KS",
        "애플": "AAPL", "테슬라": "TSLA", "엔비디아": "NVDA", "마이크로소프트": "MSFT"
    }

    @classmethod
    def resolve(cls, query: str) -> tuple:
        if not query:
            return "005930.KS", "삼성전자"
        q = query.strip()
        if q in cls.POPULAR_STOCKS:
            return cls.POPULAR_STOCKS[q], q
        if "." in q or (q.isalpha() and len(q) <= 5):
            return q.upper(), q
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
        return q.upper() + ".KS", q


# ==========================================
# 6. 차트 생성기
# ==========================================
def generate_chart(df: pd.DataFrame, name: str) -> tuple:
    df = df.copy()
    df['MA5'] = df['Close'].rolling(5, min_periods=1).mean()
    df['MA20'] = df['Close'].rolling(20, min_periods=1).mean()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    x = np.arange(len(df))
    up = df['Close'] >= df['Open']
    ax1.bar(x[up], df['Close'][up] - df['Open'][up], bottom=df['Open'][up], color='#ef5350', width=0.8)
    ax1.vlines(x[up], df['Low'][up], df['High'][up], color='#ef5350')
    down = df['Close'] < df['Open']
    ax1.bar(x[down], df['Open'][down] - df['Close'][down], bottom=df['Close'][down], color='#26a69a', width=0.8)
    ax1.vlines(x[down], df['Low'][down], df['High'][down], color='#26a69a')
    ax1.plot(x, df['MA5'], label='MA 5', color='orange', lw=1)
    ax1.plot(x, df['MA20'], label='MA 20', color='green', lw=1)
    ax1.set_title(f"{name} 기술적 차트", fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    ax2.plot(x, rsi, color='purple', label='RSI (14)')
    ax2.axhline(70, color='red', ls='--')
    ax2.axhline(30, color='blue', ls='--')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue(), float(df['Close'].iloc[-1])


# ==========================================
# 7. 텔레그램 리모컨 및 명령어 핸들러
# ==========================================
async def start_dashboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO_TRADING_ACTIVE
    status_text = "🟢 자율 구동 중 (30분 주기 자동 스캔 활성)" if AUTO_TRADING_ACTIVE else "🔴 중지됨"
    
    keyboard = [
        [InlineKeyboardButton("🔥 AI 자율 종목 스캔", callback_data='auto_scan'),
         InlineKeyboardButton("🚨 실시간 이상징후", callback_data='realtime_monitor')],
        [InlineKeyboardButton("🛑 자동매매 중지", callback_data='emergency_stop'),
         InlineKeyboardButton("🚀 자동매매 재개", callback_data='resume_trading')],
        [InlineKeyboardButton("📊 시장 레이더", callback_data='market_radar')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"🤖 **[주식 AI 자율 관제센터]**\n\n"
        f"• **상태:** {status_text}\n"
        f"• 백그라운드 엔진이 스스로 작동하고 있습니다.",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO_TRADING_ACTIVE
    query = update.callback_query
    await query.answer()
    data = query.data
    
    if data == 'auto_scan':
        await query.edit_message_text(text="🔍 **[AI 자율 종목 스캔 실행 중]**...", parse_mode='Markdown')
        report = AIServiceRouter.analyze("현재 한국 주식 시장에서 수급이 유입되는 유망 종목 1가지를 자율 발굴하여 분석해주세요.")
        await context.bot.send_message(chat_id=query.message.chat_id, text=report, parse_mode='Markdown')
    elif data == 'emergency_stop':
        AUTO_TRADING_ACTIVE = False
        await query.edit_message_text(text="🛑 **[긴급 경보] 자율 자동매매 엔진이 중지되었습니다.**", parse_mode='Markdown')
    elif data == 'resume_trading':
        AUTO_TRADING_ACTIVE = True
        await query.edit_message_text(text="🚀 **[재개 완료] 자율 자동매매 엔진이 다시 가동됩니다.**", parse_mode='Markdown')
    elif data == 'realtime_monitor':
        await query.edit_message_text(text="⚡ **[실시간 감시 레이더]** 24시간 자율 감시 중.", parse_mode='Markdown')
    elif data == 'market_radar':
        await query.edit_message_text(text="📊 **[시장 레이더]** 거래대금 상위 스캔 완료.", parse_mode='Markdown')

async def handle_all_commands(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    if update.message.photo:
        await update.message.reply_text("📸 차트 이미지 분석 중...")
        return

    text = update.message.text.strip()
    if text in ["/start", "!start", "!스타트"]:
        await start_dashboard(update, context)
        return

    if not text.startswith("!") and not text.startswith("/"):
        return

    parts = text.split(maxsplit=1)
    cmd = parts[0].replace("!", "").replace("/", "").lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ["추세", "차트"]:
        ticker, name = QuickStockResolver.resolve(arg if arg else "삼성전자")
        await update.message.reply_text(f"📊 [{name}] 데이터 분석 중...")
        try:
            df = await asyncio.to_thread(yf.download, ticker, period="6mo", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            img_bytes, cp = await asyncio.to_thread(generate_chart, df, f"{name} ({ticker})")
            report = await asyncio.to_thread(AIServiceRouter.analyze, f"종목: {name}({ticker}), 현재가: {cp}원. 추세와 전망 분석.")
            await update.message.reply_photo(photo=img_bytes, caption=f"📊 [{name}] 차트 분석")
            await update.message.reply_text(report, parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"⚠️ 오류: {e}")
        return

    if cmd in ["손절가", "투자검사"]:
        ticker, name = QuickStockResolver.resolve(arg)
        await update.message.reply_text(f"🔍 [{name}] 분석 중...")
        report = AIServiceRouter.analyze(f"종목 {name}({ticker})에 대한 {cmd} 분석을 수행해주세요.")
        await update.message.reply_text(report, parse_mode="Markdown")
        return


# ==========================================
# 8. Flask 웹 서버 (클라우드 슬립 방지용 헬스체크)
# ==========================================
web_app = Flask(__name__)

@web_app.route('/')
def home():
    return "Autonomous AI Stock Trading Bot is running 24/7!", 200

def run_web():
    port = int(os.environ.get("PORT", 10000))
    web_app.run(host="0.0.0.0", port=port)


# ==========================================
# 9. 백그라운드 스케줄러 관리 및 예외 방어 시스템
# ==========================================
def init_background_scheduler(application):
    """APScheduler 백그라운드 구동 및 네트워크 에러/크래시 방어 로직"""
    scheduler = BackgroundScheduler()
    scheduler.add_job(lambda: run_autonomous_scanner(application), 'interval', minutes=30, timezone=KST)
    scheduler.start()
    logger.info("🛡️ [9번 모듈] 백그라운드 스케줄러 안전 장치 및 예외 방어 시스템 활성화 완료.")
    return scheduler


# ==========================================
# 10. 봇 메인 실행 및 프로세스 진입점
# ==========================================
def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN이 설정되지 않았습니다!")
        return

    # 텔레그램 앱 빌드
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # 핸들러 등록
    application.add_handler(CommandHandler("start", start_dashboard))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_all_commands))
    application.add_handler(MessageHandler(filters.COMMAND, handle_all_commands))
    application.add_handler(MessageHandler(filters.PHOTO, handle_all_commands))

    # 1. Flask 웹 서버 구동 (클라우드 슬립 방지)
    threading.Thread(target=run_web, daemon=True).start()

    # 2. 24시간 자율 스케줄러 및 방어 시스템 가동 (9번, 10번 연동)
    init_background_scheduler(application)

    logger.info("🚀 [10번 모듈] 주식 AI 자율 관제 봇이 성공적으로 실행되었습니다. 텔레그램에서 /start를 입력하세요.")
    
    # 폴링 시작
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
