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

# 로깅 설정
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 0. 맷플롯립 한글 폰트 설정
# ==========================================
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. API 키 및 설정 (환경 변수 연동)
# ==========================================
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

USER_PREFERENCES = {
    "chart_style": 2,             # 1: 선차트, 2: 캔들차트
    "active_indicator": "기본"    # 기본값 설정
}


# ==========================================
# 2. 엘리어트 파동 분석 엔진
# ==========================================
class ElliottWaveAnalyzer:
    @staticmethod
    def compute_zig_zag(df: pd.DataFrame, length: int = 6):
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        n = len(df)
        
        pivots = []
        for i in range(length, n - length):
            window_high = highs[i - length : i + length + 1]
            window_low = lows[i - length : i + length + 1]
            
            if highs[i] == max(window_high):
                pivots.append({'index': i, 'price': closes[i], 'type': 'H'})
            elif lows[i] == min(window_low):
                pivots.append({'index': i, 'price': closes[i], 'type': 'L'})
                
        cleaned = []
        for p in pivots:
            if not cleaned:
                cleaned.append(p)
            else:
                if cleaned[-1]['type'] != p['type']:
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
            p0, p1, p2, p3 = subset[0]['price'], subset[1]['price'], subset[2]['price'], subset[3]['price']
            t0, t3 = subset[0]['type'], subset[3]['type']
            
            is_valid = True
            if t0 == 'L' and p1 <= p0: is_valid = False
            if is_valid and t3 == 'H' and p3 <= p1: is_valid = False
                
            pattern = []
            for idx, p in enumerate(subset):
                pattern.append({
                    'index': p['index'], 'price': p['price'], 'type': p['type'],
                    'wave': wave_sequence[idx], 'valid': is_valid
                })
            labeled_patterns.append(pattern)
        return labeled_patterns if labeled_patterns else []


# ==========================================
# 3. AI 분석 라우터
# ==========================================
class AIServiceRouter:
    @staticmethod
    def analyze(prompt: str, image_bytes: bytes = None) -> str:
        summary_instruction = (
            "[요청 사항]\n"
            "텔레그램 메시지로 전송하기 좋게 장황한 설명은 빼고, 핵심 내용만 마크다운 요약 형태로 간결하게 작성해주세요.\n"
            "포함할 내용: 1. 일목균형표/추세 요약, 2. 파동/지표 상태, 3. 추천 목표가 및 손절가, 4. 핵심 대응 전략\n\n"
        )
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
                    messages=[
                        {"role": "system", "content": "당신은 최고 수준의 금융 기술적 분석 전문가입니다."},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.2
                )
                return f"[Groq 백업 요약 분석]\n\n" + completion.choices[0].message.content
        except Exception as e:
            groq_err = str(e)

        return f"⚠️ AI 분석 실패 (Gemini: {gemini_err} / Groq: {groq_err})"


# ==========================================
# 4. 보조지표 계산 및 차트 생성 엔진
# ==========================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    
    high_9 = df['High'].rolling(window=9, min_periods=1).max()
    low_9 = df['Low'].rolling(window=9, min_periods=1).min()
    df['Tenkan_Sen'] = (high_9 + low_9) / 2
    
    high_26 = df['High'].rolling(window=26, min_periods=1).max()
    low_26 = df['Low'].rolling(window=26, min_periods=1).min()
    df['Kijun_Sen'] = (high_26 + low_26) / 2
    
    df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(26)
    
    high_52 = df['High'].rolling(window=52, min_periods=1).max()
    low_52 = df['Low'].rolling(window=52, min_periods=1).min()
    df['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)

    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=20, min_periods=1).mean()
    mad = tp.rolling(window=20, min_periods=1).apply(lambda x: np.fabs(x - x.mean()).mean(), raw=True)
    df['CCI'] = (tp - sma_tp) / (0.015 * mad + 1e-9)

    bb_std = df['Close'].rolling(window=20, min_periods=1).std()
    df['BB_Middle'] = df['MA20']
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    return df

def generate_chart_image(df: pd.DataFrame, ticker_name: str, style: int, indicator: str) -> tuple:
    is_sub_needed = indicator in ["기본", "CCI", "RSI", "MACD"]
    if is_sub_needed:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        ax2 = None

    if style == 1:
        ax1.plot(df.index, df['Close'], label='종가', color='#1f77b4', linewidth=1.8)
    else:
        up = df[df['Close'] >= df['Open']]
        down = df[df['Close'] < df['Open']]
        width = 0.85
        if not up.empty:
            ax1.bar(up.index, up['Close'] - up['Open'], width, bottom=up['Open'], color='#ef5350', alpha=0.9)
            ax1.vlines(up.index, up['Low'], up['High'], color='#ef5350', linewidth=1.2)
        if not down.empty:
            ax1.bar(down.index, down['Open'] - down['Close'], width, bottom=down['Close'], color='#26a69a', alpha=0.9)
            ax1.vlines(down.index, down['Low'], down['High'], color='#26a69a', linewidth=1.2)

    summary_text = f"종목: {ticker_name} | 현재가: {float(df['Close'].iloc[-1]):,.2f}\n"

    def draw_elliott_waves(axis):
        pivots = ElliottWaveAnalyzer.compute_zig_zag(df, length=6)
        patterns = ElliottWaveAnalyzer.validate_and_label_waves(pivots)
        if pivots:
            px = [df.index[p['index']] for p in pivots]
            py = [p['price'] for p in pivots]
            axis.plot(px, py, color='#78909c', linestyle='-', alpha=0.6, linewidth=1.2)
        if patterns:
            latest = patterns[-1]
            lx = [df.index[pt['index']] for pt in latest]
            ly = [pt['price'] for pt in latest]
            line_color = '#2962ff' if latest[0]['valid'] else '#ff5252'
            axis.plot(lx, ly, color=line_color, linewidth=2.2)
            for pt in latest:
                axis.annotate(pt['wave'], (df.index[pt['index']], pt['price']), textcoords="offset points", xytext=(0, 10 if pt['type']=='H' else -15), ha='center', fontsize=10, fontweight='bold', color=line_color)

    def draw_ichimoku_cloud(axis):
        span_a = df['Senkou_Span_A']
        span_b = df['Senkou_Span_B']
        axis.plot(df.index, span_a, label='선행스팬A', color='#00897b', linewidth=1.0, alpha=0.7)
        axis.plot(df.index, span_b, label='선행스팬B', color='#e91e63', linewidth=1.0, alpha=0.7)
        axis.fill_between(df.index, span_a, span_b, where=(span_a >= span_b), facecolor='#26a69a', alpha=0.2, interpolate=True)
        axis.fill_between(df.index, span_a, span_b, where=(span_a < span_b), facecolor='#ef5350', alpha=0.2, interpolate=True)

    if indicator == "기본":
        ax1.plot(df.index, df['MA5'], label='MA 5', color='orange', alpha=0.8, linewidth=1.2)
        ax1.plot(df.index, df['MA20'], label='MA 20', color='green', alpha=0.8, linewidth=1.2)
        draw_ichimoku_cloud(ax1)
        draw_elliott_waves(ax1)
        if ax2 is not None:
            ax2.plot(df.index, df['CCI'], label='CCI (20)', color='purple', linewidth=1.5)
            ax2.axhline(100, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
            ax2.axhline(-100, color='blue', linestyle='--', linewidth=0.8, alpha=0.7)
            ax2.set_ylabel('CCI')
            ax2.grid(True)
    elif indicator == "일목균형표":
        draw_ichimoku_cloud(ax1)
    elif indicator == "엘리어트파동":
        draw_elliott_waves(ax1)
    elif indicator == "CCI" and ax2 is not None:
        ax2.plot(df.index, df['CCI'], label='CCI (20)', color='purple', linewidth=1.5)
        ax2.axhline(100, color='red', linestyle='--')
        ax2.axhline(-100, color='blue', linestyle='--')
    elif indicator == "볼린저밴드":
        ax1.plot(df.index, df['BB_Upper'], label='상단', color='red', linestyle='--')
        ax1.plot(df.index, df['BB_Middle'], label='중단', color='blue')
        ax1.plot(df.index, df['BB_Lower'], label='하단', color='green', linestyle='--')
    elif indicator == "RSI" and ax2 is not None:
        ax2.plot(df.index, df['RSI'], label='RSI', color='teal')
        ax2.axhline(70, color='red', linestyle='--')
        ax2.axhline(30, color='blue', linestyle='--')
    elif indicator == "MACD" and ax2 is not None:
        ax2.plot(df.index, df['MACD'], label='MACD', color='blue')
        ax2.plot(df.index, df['MACD_Signal'], label='Signal', color='red')

    ax1.set_title(f"{ticker_name} 기술적 분석 ({indicator})", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.5)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue(), summary_text


# ==========================================
# 5. 네이버 증권 기반 전수 종목 자동 판별 시스템 (pykrx 대체)
# ==========================================
class UltimateStockSystem:
    def __init__(self, bot_app=None):
        self.bot_app = bot_app
        self.name_to_ticker = {}
        self.ticker_to_name = {}
        self.load_all_stocks_naver()
        
        self.scheduler = BackgroundScheduler(timezone=KST)
        self.setup_scheduler()
        
    def load_all_stocks_naver(self):
        """네이버 증권 크롤링을 통해 코스피(.KS)와 코스닥(.KQ) 전수 종목을 완벽하게 매핑"""
        try:
            for market_type, suffix in [("sise_market_sum.naver?sosok=0", ".KS"), ("sise_market_sum.naver?sosok=1", ".KQ")]:
                page = 1
                while True:
                    url = f"https://finance.naver.com/sise/{market_type}&page={page}"
                    res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                    soup = BeautifulSoup(res.text, 'html.parser')
                    rows = soup.select('table.type_2 tr')
                    
                    found = False
                    for row in rows:
                        link = row.select_one('a.tltle')
                        if link:
                            found = True
                            name = link.text.strip()
                            href = link['href']
                            code = href.split('code=')[-1]
                            full_ticker = f"{code}{suffix}"
                            self.name_to_ticker[name] = full_ticker
                            self.ticker_to_name[full_ticker] = name
                            self.name_to_ticker[code] = full_ticker
                    
                    if not found or page > 50:
                        break
                    page += 1
            logger.info(f"✅ 코스피·코스닥 전수 종목 자동 구분 매핑 완료! (총 {len(self.name_to_ticker)}개)")
        except Exception as e:
            logger.error(f"⚠️ 종목 데이터 로드 중 오류 발생: {e}")

    def resolve_ticker(self, name_or_code):
        if not name_or_code: return "005930.KS"
        q = name_or_code.strip()
        if q in self.name_to_ticker:
            return self.name_to_ticker[q]
        if "." in q:
            return q.upper()
        
        for suffix in [".KS", ".KQ"]:
            test_ticker = q.upper() + suffix
            try:
                if not yf.download(test_ticker, period="3d", progress=False).empty:
                    return test_ticker
            except:
                continue
        return q.upper() + ".KS"

    def resolve_name(self, name_or_code):
        if not name_or_code: return "삼성전자"
        q = name_or_code.strip()
        for suffix in [".KS", ".KQ"]:
            full = (q + suffix).upper()
            if full in self.ticker_to_name:
                return self.ticker_to_name[full]
        if q in self.name_to_ticker:
            return q
        return q

    def setup_scheduler(self):
        self.scheduler.add_job(self.job_0700_news, 'cron', hour=7, minute=0)
        self.scheduler.add_job(self.job_1200_comprehensive, 'cron', hour=12, minute=0)
        self.scheduler.start()

    def send_telegram_message(self, text):
        if not self.bot_app or not TELEGRAM_CHAT_ID: return
        try:
            Bot(token=TELEGRAM_BOT_TOKEN).send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
        except Exception as e:
            logger.error(f"알림 전송 실패: {e}")

    def job_0700_news(self):
        ai_brief = AIServiceRouter.analyze("오늘 장 시작 전 주목해야 할 글로벌 경제 뉴스 요약을 간결하게 제공해주세요.")
        self.send_telegram_message(f"🌅 [모닝 경제뉴스 요약]\n\n{ai_brief}")

    def job_1200_comprehensive(self):
        ai_brief = AIServiceRouter.analyze("현재 국내외 증시 상황 및 주요 관심 종목 점검 포인트를 간결하게 요약해주세요.")
        self.send_telegram_message(f"🕛 [12시 종합 투자 요약]\n\n{ai_brief}")

    def process_command(self, cmd: str, target: str) -> str:
        ticker = self.resolve_ticker(target)
        name = self.resolve_name(target)
        current_price = 0.0
        try:
            df = yf.download(ticker, period="3mo", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if not df.empty:
                current_price = float(df['Close'].iloc[-1])
        except Exception:
            pass

        if cmd in ["!손절가", "/손절가"]:
            s1 = current_price * 0.97 if current_price else 0
            s2 = current_price * 0.94 if current_price else 0
            s3 = current_price * 0.90 if current_price else 0
            ai_comment = AIServiceRouter.analyze(f"{name} 종목의 현재가 {current_price}원 기준 핵심 손절 기준을 요약해주세요.")
            return f"🛡️ [{name}] ({ticker}) 3단계 손절가 산출\n- 현재가: {current_price:,.2f}\n- 1차 손절선 (-3%): {s1:,.2f}\n- 2차 손절선 (-6%): {s2:,.2f}\n- 3차 손절선 (-10%): {s3:,.2f}\n\n{ai_comment}"

        elif cmd in ["!투자조언", "/투자조언"]:
            return AIServiceRouter.analyze(f"{name} ({ticker}) 종목(현재가: {current_price:,.2f}원)의 핵심 투자 전략과 포인트를 요약해주세요.")

        elif cmd in ["!뉴스분석", "/뉴스분석"]:
            return AIServiceRouter.analyze(f"{name} ({ticker}) 종목 관련 최근 시장 이슈 및 업황을 핵심 위주로 요약해주세요.")

        elif cmd in ["!명령어", "/명령어", "!help", "/help", "/start"]:
            return (
                "📌 [전체 명령어 및 기능 안내]\n\n"
                "1. !추세 [종목명/코드] : 코스피/코스닥 전수 자동 구분 차트 및 AI 분석\n"
                "2. !손절가 [종목명/코드] : 3단계 손절선 자동 산출\n"
                "3. !투자조언 [종목명/코드] : AI 기반 핵심 투자 전략 제공\n"
                "4. !뉴스분석 [종목명/코드] : 최신 업황 및 뉴스 요약 리포트\n"
                "5. !차트변경 [1 또는 2] : 1(선차트) / 2(캔들차트)\n"
                f"6. !보조지표 [{' / '.join(AVAILABLE_INDICATORS)}] : 지표 설정\n"
                "7. !명령어 : 전체 명령어 목록 확인"
            )
        return AIServiceRouter.analyze(f"종목명: {name} ({ticker}), 현재가: {current_price}원에 대한 핵심 투자 분석을 제공해주세요.")


# ==========================================
# 6. 텔레그램 핸들러 및 봇 실행
# ==========================================
app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
stock_system = UltimateStockSystem(bot_app=app)

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    text = update.message.text.strip()

    if text.startswith("!") or text.startswith("/"):
        parts = text.split(maxsplit=1)
        cmd = parts[0]
        target = parts[1] if len(parts) > 1 else None

        if cmd in ["!차트변경", "/차트변경"]:
            if target in ["1", "2"]:
                USER_PREFERENCES["chart_style"] = int(target)
                await update.message.reply_text(f"✅ 차트 스타일 변경 완료 ({'선차트' if target=='1' else '캔들차트'})")
            else:
                await update.message.reply_text("⚠️ 사용법: !차트변경 1 또는 !차트변경 2")
            return

        if cmd in ["!보조지표", "/보조지표"]:
            if target in AVAILABLE_INDICATORS:
                USER_PREFERENCES["active_indicator"] = target
                await update.message.reply_text(f"✅ 보조지표 설정 완료: [{target}]")
            else:
                await update.message.reply_text(f"⚠️ 지원 지표: {', '.join(AVAILABLE_INDICATORS)}")
            return

        if cmd in ["!추세", "/추세"]:
            ticker = stock_system.resolve_ticker(target)
            name = stock_system.resolve_name(target)
            await update.message.reply_text(f"📊 [{name}] ({ticker}) 차트 생성 및 AI 분석 중...")
            
            try:
                df = yf.download(ticker, period="6mo", interval="1d", progress=False)
                if df.empty:
                    alt_ticker = ticker.replace(".KS", ".KQ") if ticker.endswith(".KS") else ticker.replace(".KQ", ".KS")
                    df = yf.download(alt_ticker, period="6mo", interval="1d", progress=False)
                    if not df.empty: ticker = alt_ticker

                if df.empty:
                    await update.message.reply_text("❌ 해당 종목의 데이터를 찾을 수 없습니다.")
                    return

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                df = calculate_indicators(df)
                img_bytes, summary = generate_chart_image(df, f"{name} ({ticker})", USER_PREFERENCES["chart_style"], USER_PREFERENCES["active_indicator"])
                ai_report = AIServiceRouter.analyze(f"차트 및 기술적 분석 요약:\n\n{summary}", img_bytes)
                
                await update.message.reply_photo(photo=img_bytes, caption=f"📊 [{name}] 기술적 분석 차트")
                if len(ai_report) > 4000:
                    for i in range(0, len(ai_report), 4000):
                        await update.message.reply_text(ai_report[i:i+4000])
                else:
                    await update.message.reply_text(ai_report)
            except Exception as e:
                await update.message.reply_text(f"⚠️ 오류 발생: {e}")
            return
        else:
            result = stock_system.process_command(cmd, target)
            await update.message.reply_text(result)
            return

if __name__ == '__main__':
    app.add_handler(MessageHandler(filters.TEXT, handle_text_message))
    print("🤖 대한민국 코스피·코스닥 전수 종목 자동 판별 텔레그램 봇이 가동되었습니다.")
    app.run_polling()
