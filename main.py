import os
import io
import logging
import platform
from datetime import datetime
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
import google.generativeai as genai
from groq import Groq
from flask import Flask
import threading
import asyncio

# 로깅 설정
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# API 키 및 환경 변수
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

KST = pytz.timezone('Asia/Seoul')
AVAILABLE_INDICATORS = ["기본", "일목균형표", "엘리어트파동", "CCI", "볼린저밴드", "RSI", "MACD"]
USER_PREFERENCES = {"chart_style": 2, "active_indicator": "기본"}

# 엘리어트 파동 분석 클래스
class ElliottWaveAnalyzer:
    @staticmethod
    def compute_zig_zag(df: pd.DataFrame, length: int = 6):
        highs, lows, closes = df['High'].values, df['Low'].values, df['Close'].values
        n = len(df)
        pivots = []
        for i in range(length, n - length):
            if highs[i] == max(highs[i - length : i + length + 1]):
                pivots.append({'index': i, 'price': closes[i], 'type': 'H'})
            elif lows[i] == min(lows[i - length : i + length + 1]):
                pivots.append({'index': i, 'price': closes[i], 'type': 'L'})
        cleaned = []
        for p in pivots:
            if not cleaned or cleaned[-1]['type'] != p['type']:
                cleaned.append(p)
            else:
                if p['type'] == 'H' and p['price'] > cleaned[-1]['price']:
                    cleaned[-1] = p
                elif p['type'] == 'L' and p['price'] < cleaned[-1]['price']:
                    cleaned[-1] = p
        return cleaned

    @staticmethod
    def validate_and_label_waves(pivots):
        wave_sequence = ['1', '2', '3', '4', '5', 'A', 'B', 'C']
        labeled_patterns = []
        for i in range(len(pivots) - 7):
            subset = pivots[i:i+8]
            p0, p1, p3 = subset[0]['price'], subset[1]['price'], subset[3]['price']
            t0, t3 = subset[0]['type'], subset[3]['type']
            is_valid = not (t0 == 'L' and p1 <= p0) and not (t3 == 'H' and p3 <= p1)
            pattern = [{'index': p['index'], 'price': p['price'], 'type': p['type'], 'wave': wave_sequence[idx], 'valid': is_valid} for idx, p in enumerate(subset)]
            labeled_patterns.append(pattern)
        return labeled_patterns if labeled_patterns else []

# AI 분석 라우터
class AIServiceRouter:
    @staticmethod
    def analyze(prompt: str, image_bytes: bytes = None) -> str:
        summary_instruction = "[요청 사항]\n장황한 설명은 빼고 핵심 내용만 마크다운 요약 형태로 간결하게 작성해주세요.\n\n"
        full_prompt = summary_instruction + prompt
        gemini_err, groq_err = "", ""
        try:
            if GEMINI_API_KEY:
                model = genai.GenerativeModel('gemini-1.5-flash')
                content = [full_prompt, {'mime_type': 'image/png', 'data': image_bytes}] if image_bytes else [full_prompt]
                response = model.generate_content(content)
                if response.text:
                    return f"[Gemini AI 요약 분석]\n\n" + response.text
        except Exception as e:
            gemini_err = str(e)
        try:
            if groq_client:
                completion = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": "당신은 금융 기술적 분석 전문가입니다."}, {"role": "user", "content": full_prompt}],
                    temperature=0.2
                )
                return f"[Groq 백업 요약 분석]\n\n" + completion.choices[0].message.content
        except Exception as e:
            groq_err = str(e)
        return f"⚠️ AI 분석 실패 (Gemini: {gemini_err} / Groq: {groq_err})"

# 보조지표 및 차트 생성 함수
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    high_9, low_9 = df['High'].rolling(9, min_periods=1).max(), df['Low'].rolling(9, min_periods=1).min()
    df['Tenkan_Sen'] = (high_9 + low_9) / 2
    high_26, low_26 = df['High'].rolling(26, min_periods=1).max(), df['Low'].rolling(26, min_periods=1).min()
    df['Kijun_Sen'] = (high_26 + low_26) / 2
    df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(26)
    high_52, low_52 = df['High'].rolling(52, min_periods=1).max(), df['Low'].rolling(52, min_periods=1).min()
    df['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(20, min_periods=1).mean()
    mad = tp.rolling(20, min_periods=1).apply(lambda x: np.fabs(x - x.mean()).mean(), raw=True)
    df['CCI'] = (tp - sma_tp) / (0.015 * mad + 1e-9)
    bb_std = df['Close'].rolling(20, min_periods=1).std()
    df['BB_Middle'], df['BB_Upper'], df['BB_Lower'] = df['MA20'], df['MA20'] + (bb_std * 2), df['MA20'] - (bb_std * 2)
    delta = df['Close'].diff()
    gain, loss = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean(), (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    ema12, ema26 = df['Close'].ewm(span=12, adjust=False).mean(), df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'], df['MACD_Signal'] = ema12 - ema26, (ema12 - ema26).ewm(span=9, adjust=False).mean()
    return df

def generate_chart_image(df: pd.DataFrame, ticker_name: str, style: int, indicator: str) -> tuple:
    df = df.copy()
    date_strings = df.index.strftime('%Y-%m-%d').values
    x_indexes = np.arange(len(df))
    is_sub_needed = indicator in ["기본", "CCI", "RSI", "MACD"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True) if is_sub_needed else (plt.subplots(1, 1, figsize=(10, 6)), None)

    if style == 1:
        ax1.plot(x_indexes, df['Close'].values, label='종가', color='#1f77b4', linewidth=1.8)
    else:
        up_mask, down_mask = df['Close'] >= df['Open'], df['Close'] < df['Open']
        if len(x_indexes[up_mask]) > 0:
            ax1.bar(x_indexes[up_mask], df['Close'].values[up_mask] - df['Open'].values[up_mask], 0.8, bottom=df['Open'].values[up_mask], color='#ef5350', alpha=0.9)
            ax1.vlines(x_indexes[up_mask], df['Low'].values[up_mask], df['High'].values[up_mask], color='#ef5350', linewidth=1.2)
        if len(x_indexes[down_mask]) > 0:
            ax1.bar(x_indexes[down_mask], df['Open'].values[down_mask] - df['Close'].values[down_mask], 0.8, bottom=df['Close'].values[down_mask], color='#26a69a', alpha=0.9)
            ax1.vlines(x_indexes[down_mask], df['Low'].values[down_mask], df['High'].values[down_mask], color='#26a69a', linewidth=1.2)

    summary_text = f"종목: {ticker_name} | 현재가: {float(df['Close'].iloc[-1]):,.2f}\n"

    def draw_ichimoku(axis):
        sa, sb = df['Senkou_Span_A'].values, df['Senkou_Span_B'].values
        axis.plot(x_indexes, sa, color='#00897b', alpha=0.7)
        axis.plot(x_indexes, sb, color='#e91e63', alpha=0.7)
        axis.fill_between(x_indexes, sa, sb, where=(sa >= sb), facecolor='#26a69a', alpha=0.2, interpolate=True)
        axis.fill_between(x_indexes, sa, sb, where=(sa < sb), facecolor='#ef5350', alpha=0.2, interpolate=True)

    if indicator == "기본":
        ax1.plot(x_indexes, df['MA5'].values, label='MA 5', color='orange')
        ax1.plot(x_indexes, df['MA20'].values, label='MA 20', color='green')
        draw_ichimoku(ax1)
        if ax2 is not None:
            ax2.plot(x_indexes, df['CCI'].values, color='purple')
            ax2.axhline(100, color='red', linestyle='--')
            ax2.axhline(-100, color='blue', linestyle='--')
    elif indicator == "일목균형표": draw_ichimoku(ax1)
    elif indicator == "볼린저밴드":
        ax1.plot(x_indexes, df['BB_Upper'].values, color='red', linestyle='--')
        ax1.plot(x_indexes, df['BB_Middle'].values, color='blue')
        ax1.plot(x_indexes, df['BB_Lower'].values, color='green', linestyle='--')
    elif indicator == "RSI" and ax2 is not None:
        ax2.plot(x_indexes, df['RSI'].values, color='teal')
        ax2.axhline(70, color='red', linestyle='--')
        ax2.axhline(30, color='blue', linestyle='--')

    step = max(1, len(x_indexes) // 6)
    (ax2 if ax2 is not None else ax1).set_xticks(x_indexes[::step])
    (ax2 if ax2 is not None else ax1).set_xticklabels(date_strings[::step])
    ax1.set_title(f"{ticker_name} 기술적 분석 ({indicator})", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.5)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue(), summary_text

# 주식 정보 시스템 및 스케줄러
class UltimateStockSystem:
    def __init__(self, bot_app=None):
        self.bot_app = bot_app
        self.name_to_ticker, self.ticker_to_name = {}, {}
        self.load_all_stocks()
        self.scheduler = BackgroundScheduler(timezone=KST)
        self.scheduler.add_job(self.job_0700_news, 'cron', hour=7, minute=0)
        self.scheduler.start()

    def load_all_stocks(self):
        try:
            for market_type, suffix in [("sise_market_sum.naver?sosok=0", ".KS"), ("sise_market_sum.naver?sosok=1", ".KQ")]:
                for page in range(1, 10):
                    res = requests.get(f"https://finance.naver.com/sise/{market_type}&page={page}", headers={'User-Agent': 'Mozilla/5.0'})
                    soup = BeautifulSoup(res.text, 'html.parser')
                    rows = soup.select('table.type_2 tr')
                    found = False
                    for row in rows:
                        link = row.select_one('a.tltle')
                        if link:
                            found = True
                            name, code = link.text.strip(), link['href'].split('code=')[-1]
                            full_ticker = f"{code}{suffix}"
                            self.name_to_ticker[name], self.ticker_to_name[full_ticker], self.name_to_ticker[code] = full_ticker, name, full_ticker
                    if not found: break
        except Exception as e:
            logger.error(f"종목 로드 오류: {e}")

    def resolve_ticker(self, q):
        if not q: return "005930.KS"
        q = q.strip()
        if q in self.name_to_ticker: return self.name_to_ticker[q]
        return q.upper() + (".KS" if "." not in q else "")

    def resolve_name(self, q):
        if not q: return "삼성전자"
        q = q.strip()
        for suffix in [".KS", ".KQ"]:
            if (q + suffix).upper() in self.ticker_to_name: return self.ticker_to_name[(q + suffix).upper()]
        return q

    def send_telegram_message(self, text):
        if not self.bot_app or not TELEGRAM_CHAT_ID: return
        async def send(): await self.bot_app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
        try: asyncio.get_running_loop().create_task(send())
        except RuntimeError: pass

    def job_0700_news(self):
        self.send_telegram_message(f"🌅 [모닝 뉴스 요약]\n\n{AIServiceRouter.analyze('오늘 장 시작 전 글로벌 경제 뉴스 요약')}")

    def process_command(self, cmd: str, target: str) -> str:
        ticker, name = self.resolve_ticker(target), self.resolve_name(target)
        if cmd in ["!손절가", "/손절가"]:
            df = yf.download(ticker, period="3d", progress=False)
            cp = float(df['Close'].iloc[-1]) if not df.empty else 0.0
            return f"🛡️ [{name}] 손절가\n- 1차(-3%): {cp*0.97:,.2f}\n- 2차(-6%): {cp*0.94:,.2f}"
        return AIServiceRouter.analyze(f"{name} ({ticker}) 핵심 투자 분석 제공")

# 텔레그램 메시지 핸들러
async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    text = update.message.text.strip()
    if text.startswith("!") or text.startswith("/"):
        parts = text.split(maxsplit=1)
        cmd, target = parts[0], parts[1] if len(parts) > 1 else None
        if cmd in ["!추세", "/추세"]:
            ticker, name = stock_system.resolve_ticker(target), stock_system.resolve_name(target)
            await update.message.reply_text(f"📊 [{name}] 분석 중...")
            try:
                df = yf.download(ticker, period="6mo", interval="1d", progress=False)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                df = calculate_indicators(df)
                img_bytes, summary = generate_chart_image(df, f"{name} ({ticker})", USER_PREFERENCES["chart_style"], USER_PREFERENCES["active_indicator"])
                ai_report = AIServiceRouter.analyze(f"차트 요약:\n\n{summary}", img_bytes)
                await update.message.reply_photo(photo=img_bytes, caption=f"📊 [{name}] 분석 차트")
                await update.message.reply_text(ai_report)
            except Exception as e:
                await update.message.reply_text(f"⚠️ 오류: {e}")
        else:
            await update.message.reply_text(stock_system.process_command(cmd, target))

# Flask 웹 서버 (렌더 생존용)
web_app = Flask(__name__)
@web_app.route('/')
def home(): return "Bot is running live!"

def run_web():
    web_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text_message))
    app.add_handler(MessageHandler(filters.COMMAND, handle_text_message))
    stock_system = UltimateStockSystem(bot_app=app)
    
    # 웹 서버 백그라운드 구동
    threading.Thread(target=run_web, daemon=True).start()
    app.run_polling()
    
    print("🤖 대한민국 코스피·코스닥 전수 종목 자동 판별 텔레그램 봇이 가동되었습니다.")
    app.run_polling()
