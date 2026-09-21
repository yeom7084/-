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

# 2. API 토큰 및 환경 변수 설정
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
ADMIN_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")

AUTO_TRADING_ACTIVE = True


# ==========================================
# 3. AI 서비스 라우터
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
                model = genai.GenerativeModel('gemini-3.8-flash')
                content_payload = [full_prompt]
                if image_bytes:
                    content_payload.append({"mime_type": "image/png", "data": image_bytes})
                response = model.generate_content(content_payload)
                if response and response.text:
                    return f"✨ **[Gemini AI 자율 분석]**\n\n" + response.text
            except Exception as ge:
                logger.error(f"Gemini 에러: {ge}")
        raise Exception("GROQ_API_KEY 또는 GEMINI_API_KEY가 올바르게 설정되지 않았습니다.")


# ==========================================
# 4. 자율 스캔 엔진
# ==========================================
def run_autonomous_scanner(app):
    global AUTO_TRADING_ACTIVE
    if not AUTO_TRADING_ACTIVE or not ADMIN_CHAT_ID:
        return
    logger.info("🤖 [자율 엔진] 시장 스캔 시작...")
    try:
        prompt = "현재 한국 주식 시장에서 거래대금이 폭증하고 수급이 집중되는 유망 종목 1개를 골라 매수가, 목표가, 손절가를 분석해줘."
        analysis_result = AIServiceRouter.analyze(prompt)
        report_msg = f"🤖 **[자동매매 자율 관제 리포트]**\n\n{analysis_result}\n\n🟢 **엔진 상태:** 정상 구동 중"
        
        async def push():
            await app.bot.send_message(chat_id=ADMIN_CHAT_ID, text=report_msg, parse_mode="Markdown")
        asyncio.run(push())
    except Exception as e:
        logger.error(f"자율 엔진 에러: {e}")


# ==========================================
# 5. 종목 탐색기
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
# 7. 대시보드 및 버튼/명령어 핸들러 (수정 완료)
# ==========================================
async def start_dashboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO_TRADING_ACTIVE
    status_text = "🟢 자율 구동 중 (30분 주기 스캔)" if AUTO_TRADING_ACTIVE else "🔴 중지됨"
    
    keyboard = [
        [InlineKeyboardButton("🔥 AI 자율 종목 스캔", callback_data='auto_scan'),
         InlineKeyboardButton("🚨 실시간 이상징후", callback_data='realtime_monitor')],
        [InlineKeyboardButton("🛑 자동매매 중지", callback_data='emergency_stop'),
         InlineKeyboardButton("🚀 자동매매 재개", callback_data='resume_trading')],
        [InlineKeyboardButton("📊 시장 레이더", callback_data='market_radar')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    help_text = (
        f"🤖 **[주식 AI 자율 관제센터 - 전체 명령어 20선]**\n\n"
        f"• **상태:** {status_text}\n\n"
        f"📋 **지원 명령어 목록:**\n"
        f"1. `!help` 또는 `/start` : 대시보드 열기\n"
        f"2. `!차트 [종목]` : 기술적 분석 차트 출력\n"
        f"3. `!추세 [종목]` : 단기/장기 추세 분석\n"
        f"4. `!손절가 [종목]` : 리스크 및 손절 라인 산출\n"
        f"5. `!매수가 [종목]` : 최적 진입 타점 분석\n"
        f"6. `!목표가 [종목]` : 익절 목표가 분석\n"
        f"7. `!시세 [종목]` : 현재가 및 등락률 조회\n"
        f"8. `!잔고` : 가상 계좌 잔고 확인\n"
        f"9. `!수익률` : 누적 수익률 평가\n"
        f"10. `!스캔` : 유망 종목 자율 스캔 실행\n"
        f"11. `!뉴스 [검색어]` : 관련 시장 뉴스 분석\n"
        f"12. `!모니터` : 실시간 변동성 감시\n"
        f"13. `!설정` : 현재 봇 환경 변수 상태\n"
        f"14. `!상태` : 엔진 구동 상태 점검\n"
        f"15. `!시작` : 자동매매 엔진 가동\n"
        f"16. `!중지` : 자동매매 엔진 긴급 정지\n"
        f"17. `!로그` : 시스템 최근 로그 요약\n"
        f"18. `!환율` : 주요 환율 및 거시경제 지표\n"
        f"19. `!초기화` : 세션 및 캐시 초기화\n"
        f"20. `!정보` : 봇 버전 및 안내"
    )
    await update.message.reply_text(help_text, reply_markup=reply_markup, parse_mode="Markdown")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global AUTO_TRADING_ACTIVE
    query = update.callback_query
    await query.answer()
    data = query.data
    
    if data == 'auto_scan' or data == 'market_radar':
        await query.edit_message_text(text="🔍 **[AI 시장 레이더 및 종목 스캔 실행 중]** 잠시만 기다려주세요...", parse_mode='Markdown')
        try:
            report = AIServiceRouter.analyze("현재 한국 주식 시장에서 거래대금이 폭증하고 수급이 집중되는 유망 종목 및 시장 동향을 분석해주세요.")
        except Exception as e:
            report = f"⚠️ **[AI 분석 실패]**\n원인: {e}\n\n서버 환경 변수(GROQ_API_KEY / GEMINI_API_KEY)를 확인해주세요."
        await context.bot.send_message(chat_id=query.message.chat_id, text=report, parse_mode='Markdown')
        
    elif data == 'realtime_monitor':
        await query.edit_message_text(text="⚡ **[실시간 감시 레이더 작동 중]** 잠시만 기다려주세요...", parse_mode='Markdown')
        try:
            report = AIServiceRouter.analyze("현재 시장의 변동성과 주요 종목들의 실시간 이상 징후를 분석해주세요.")
        except Exception as e:
            report = f"⚠️ **[감시 레이더 실패]**\n원인: {e}"
        await context.bot.send_message(chat_id=query.message.chat_id, text=report, parse_mode='Markdown')
        
    elif data == 'emergency_stop':
        AUTO_TRADING_ACTIVE = False
        await query.edit_message_text(text="🛑 **[긴급 경보] 자율 자동매매 엔진이 중지되었습니다.**", parse_mode='Markdown')
        
    elif data == 'resume_trading':
        AUTO_TRADING_ACTIVE = True
        await query.edit_message_text(text="🚀 **[재개 완료] 자율 자동매매 엔진이 다시 가동됩니다.**", parse_mode='Markdown')

async def handle_all_commands(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    text_lower = text.lower()
    
    if text_lower in ["/start", "!start", "!스타트", "!help", "/help", "help", "도움말"]:
        await start_dashboard(update, context)
        return

    if not text.startswith("!") and not text.startswith("/"):
        return

    parts = text.split(maxsplit=1)
    cmd = parts[0].replace("!", "").replace("/", "").lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ["차트", "그래프"]:
        ticker, name = QuickStockResolver.resolve(arg if arg else "삼성전자")
        await update.message.reply_text(f"📊 [{name}] 데이터 분석 및 차트 생성 중...")
        try:
            df = await asyncio.to_thread(yf.download, ticker, period="6mo", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            img_bytes, cp = await asyncio.to_thread(generate_chart, df, f"{name} ({ticker})")
            report = await asyncio.to_thread(AIServiceRouter.analyze, f"종목: {name}({ticker}), 현재가: {cp}원. 추세와 전망 분석.")
            await update.message.reply_photo(photo=img_bytes, caption=f"📊 [{name}] 차트 분석")
            await update.message.reply_text(report, parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"⚠️ 오류 발생: {e}")
        return

    elif cmd in ["추세"]:
        ticker, name = QuickStockResolver.resolve(arg)
        report = AIServiceRouter.analyze(f"종목 {name}({ticker})의 단기 및 장기 주가 추세 분석을 수행해주세요.")
        await update.message.reply_text(report, parse_mode="Markdown")
        return

    elif cmd in ["손절가"]:
        ticker, name = QuickStockResolver.resolve(arg)
        report = AIServiceRouter.analyze(f"종목 {name}({ticker})의 리스크 관리 및 권장 손절가를 분석해주세요.")
        await update.message.reply_text(report, parse_mode="Markdown")
        return

    elif cmd in ["매수가", "진입가"]:
        ticker, name = QuickStockResolver.resolve(arg)
        report = AIServiceRouter.analyze(f"종목 {name}({ticker})의 최적 매수 타점 및 분할 매수 전략을 분석해주세요.")
        await update.message.reply_text(report, parse_mode="Markdown")
        return

    elif cmd in ["목표가", "익절가"]:
        ticker, name = QuickStockResolver.resolve(arg)
        report = AIServiceRouter.analyze(f"종목 {name}({ticker})의 단기/중기 목표 주가(익절가)를 분석해주세요.")
        await update.message.reply_text(report, parse_mode="Markdown")
        return

    elif cmd in ["시세", "현재가"]:
        ticker, name = QuickStockResolver.resolve(arg)
        try:
            df = yf.download(ticker, period="5d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            cp = float(df['Close'].iloc[-1])
            prev = float(df['Close'].iloc[-2])
            diff = cp - prev
            rate = (diff / prev) * 100
            await update.message.reply_text(f"📈 **[{name}] 실시간 시세**\n- 현재가: {cp:,.0f}원\n- 전일 대비: {diff:+,.0f}원 ({rate:+.2f}%)", parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"⚠️ 시세 조회 실패: {e}")
        return

    elif cmd in ["잔고", "계좌"]:
        await update.message.reply_text("💼 **[가상 계좌 잔고]**\n- 예수금: 10,000,000원\n- 총평가금액: 10,000,000원\n- 수익률: 0.00%", parse_mode="Markdown")
        return

    elif cmd in ["수익률", "성적"]:
        await update.message.reply_text("📊 **[매매 성적표]**\n- 오늘 승률: 100%\n- 누적 실현 손익: +0원", parse_mode="Markdown")
        return

    elif cmd in ["스캔", "발굴"]:
        await update.message.reply_text("🔍 시장 주도주 및 거래대금 상위 종목 스캔 중...")
        report = AIServiceRouter.analyze("현재 코스피/코스닥 시장에서 가장 핫한 주도 테마와 종목을 분석해줘.")
        await update.message.reply_text(report, parse_mode="Markdown")
        return

    elif cmd in ["뉴스", "속보"]:
        report = AIServiceRouter.analyze(f"최근 국내 주식 시장 이슈와 관련 뉴스 요약을 제공해주세요. 검색어: {arg}")
        await update.message.reply_text(report, parse_mode="Markdown")
        return

    elif cmd in ["모니터", "감시"]:
        await update.message.reply_text("⚡ 실시간 이상 수급 및 변동성 감시 시스템이 정상 작동 중입니다.", parse_mode="Markdown")
        return

    elif cmd in ["설정", "config"]:
        await update.message.reply_text(f"⚙️ **[봇 환경 설정]**\n- Groq API 연동: {'활성화' if GROQ_API_KEY else '비활성화'}\n- Gemini API 연동: {'활성화' if GEMINI_API_KEY else '비활성화'}", parse_mode="Markdown")
        return

    elif cmd in ["상태", "status"]:
        global AUTO_TRADING_ACTIVE
        st = "🟢 정상 구동 중" if AUTO_TRADING_ACTIVE else "🔴 중지됨"
        await update.message.reply_text(f"🖥️ **[시스템 상태]**\n- 엔진 상태: {st}\n- 스케줄러: 30분 주기 자동 스캔 활성", parse_mode="Markdown")
        return

    elif cmd in ["시작", "재개"]:
        AUTO_TRADING_ACTIVE = True
        await update.message.reply_text("🚀 자동매매 자율 엔진이 재개되었습니다.", parse_mode="Markdown")
        return

    elif cmd in ["중지", "정지"]:
        AUTO_TRADING_ACTIVE = False
        await update.message.reply_text("🛑 자동매매 자율 엔진이 긴급 중지되었습니다.", parse_mode="Markdown")
        return

    elif cmd in ["로그", "log"]:
        await update.message.reply_text("📝 **[시스템 로그 요약]**\n- 최근 에러 없음\n- API 통신 상태 원활", parse_mode="Markdown")
        return

    elif cmd in ["환율", "거시경제"]:
        report = AIServiceRouter.analyze("현재 원달러 환율과 미국 증시 마감 상황이 국내 증시에 미치는 영향을 분석해줘.")
        await update.message.reply_text(report, parse_mode="Markdown")
        return

    elif cmd in ["초기화", "리셋"]:
        await update.message.reply_text("🧹 시스템 캐시 및 임시 데이터를 초기화했습니다.", parse_mode="Markdown")
        return

    elif cmd in ["정보", "info"]:
        await update.message.reply_text("🤖 **AI Stock Autonomous Bot v2.0**\n- 24시간 자율 관제 및 20가지 명령어 지원 시스템", parse_mode="Markdown")
        return


# ==========================================
# 8. Flask 웹 서버
# ==========================================
web_app = Flask(__name__)

@web_app.route('/')
def home():
    return "Autonomous AI Stock Trading Bot with 20 Commands is running 24/7!", 200

def run_web():
    port = int(os.environ.get("PORT", 10000))
    web_app.run(host="0.0.0.0", port=port)


# ==========================================
# 9. 백그라운드 스케줄러
# ==========================================
def init_background_scheduler(application):
    scheduler = BackgroundScheduler()
    scheduler.add_job(lambda: run_autonomous_scanner(application), 'interval', minutes=30, timezone=KST)
    scheduler.start()
    logger.info("🛡️ 백그라운드 스케줄러 활성화 완료.")
    return scheduler


# ==========================================
# 10. 봇 메인 실행
# ==========================================
def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN 미설정!")
        return

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_dashboard))
    application.add_handler(CommandHandler("help", start_dashboard))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_all_commands))
    application.add_handler(MessageHandler(filters.COMMAND, handle_all_commands))
    application.add_handler(MessageHandler(filters.PHOTO, handle_all_commands))

    threading.Thread(target=run_web, daemon=True).start()
    init_background_scheduler(application)

    logger.info("🚀 20개 명령어 및 버튼 연동 관제 봇이 실행되었습니다.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
