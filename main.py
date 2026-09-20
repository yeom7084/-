import os
import io
import logging
import platform
import threading
import asyncio
from flask import Flask
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
    MessageHandler,
    ContextTypes,
    filters
)
import google.generativeai as genai
from openai import OpenAI

# 1. 로깅 설정
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

# 2. API 키 및 클라이언트 설정
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

GEMINI_API_KEY = (
    os.environ.get("GEMINI_API_KEY") or 
    os.environ.get("GOOGLE_API_KEY") or 
    os.environ.get("GEMINI_KEY") or ""
).strip()

# openai/gpt-oss-120b 또는 일반 Groq/OpenAI 호환 API 키 (OpenRouter, Groq 등)
GPT_OSS_API_KEY = (
    os.environ.get("GPT_OSS_API_KEY") or 
    os.environ.get("OPENAI_API_KEY") or 
    os.environ.get("GROQ_API_KEY") or ""
).strip()

GPT_OSS_BASE_URL = os.environ.get("GPT_OSS_BASE_URL", "https://openrouter.ai/api/v1")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

gpt_oss_client = OpenAI(
    api_key=GPT_OSS_API_KEY,
    base_url=GPT_OSS_BASE_URL
) if GPT_OSS_API_KEY else None


# ==========================================
# 3. 종목 코드 판별기
# ==========================================
class QuickStockResolver:
    KNOWN_STOCKS = {
        "삼성전자": "005930.KS", "SK하이닉스": "000660.KS", "태웅": "044780.KQ", 
        "에코프로": "086520.KQ", "에코프로비엠": "247540.KS", "셀트리온": "068270.KS",
        "LG에너지솔루션": "373220.KS", "현대차": "005380.KS", "기아": "000270.KS"
    }

    @classmethod
    def resolve(cls, query: str) -> tuple:
        if not query:
            return "005930.KS", "삼성전자"
        q = query.strip()
        if q in cls.KNOWN_STOCKS:
            return cls.KNOWN_STOCKS[q], q
        if "." in q:
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
# 4. AI 분석 라우터 (gpt-oss-120b & gemini-3.8-flash 연동)
# ==========================================
class AIServiceRouter:
    @staticmethod
    def analyze(prompt: str, image_bytes: bytes = None) -> str:
        sys_instruction = (
            "당신은 냉철하고 전문적인 주식 시장 분석가입니다.\n"
            "단순히 '오를 것 같다'고 결론내리지 말고, [뉴스 -> 사업 -> 실적 -> 수급 -> 차트 -> 시장 기대 -> 현재 주가 반영 -> 리스크] "
            "순서로 확인하고 긍정적인 근거와 부정적인 근거를 균형 있게 마크다운 요약 형태로 작성해주세요.\n\n"
        )
        full_prompt = sys_instruction + prompt
        
        gpt_error = ""
        gemini_error = ""

        # 1차 시도: openai/gpt-oss-120b (고성능 추론 모델)
        try:
            if gpt_oss_client:
                comp = gpt_oss_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0.2,
                    timeout=25
                )
                if comp.choices and comp.choices[0].message.content:
                    return f"⚡ **[gpt-oss-120b 심층 분석 리포트]**\n\n" + comp.choices[0].message.content
        except Exception as e:
            gpt_error = str(e)
            logger.warning(f"gpt-oss-120b 호출 오류 (Gemini 3.8 Flash로 전환 시도): {e}")

        # 2차 시도: gemini-3.8-flash (고속 멀티모달 플래시 모델)
        try:
            if GEMINI_API_KEY:
                model = genai.GenerativeModel('gemini-3.8-flash')
                content = [full_prompt, {'mime_type': 'image/png', 'data': image_bytes}] if image_bytes else [full_prompt]
                res = model.generate_content(content)
                if res and res.text:
                    return f"🤖 **[gemini-3.8-flash 분석 리포트]**\n\n" + res.text
        except Exception as e:
            gemini_error = str(e)
            logger.warning(f"gemini-3.8-flash 호출 오류: {e}")

        return (
            f"⚠️ **모든 AI 분석 엔진 호출에 실패했습니다.**\n\n"
            f"• **gpt-oss-120b 오류:** `{gpt_error or '키 없음 또는 응답 지연'}`\n"
            f"• **gemini-3.8-flash 오류:** `{gemini_error or '키 없음 또는 응답 지연'}`\n"
        )


# ==========================================
# 5. 차트 및 기술적 지표 생성기
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
    ax1.set_title(f"{name} 기술적 차트 분석", fontweight='bold')
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
# 6. 전체 명령어 통합 핸들러
# ==========================================
async def handle_all_commands(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    
    text = update.message.text.strip()
    if not text.startswith("!") and not text.startswith("/"):
        return

    parts = text.split(maxsplit=1)
    cmd = parts[0].replace("!", "").replace("/", "").lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ["설명", "help", "스타트", "start"]:
        await update.message.reply_text(
            "🤖 **[주식 종합 분석 봇 명령어 가이드]**\n\n"
            "📰 **뉴스 및 실적 분석**\n"
            "• `!뉴스분석 [링크 또는 내용]`\n"
            "• `!뉴스기간 [종목] [기간]`\n"
            "• `!실적발표 [종목]`\n"
            "• `!저평가` / `!서프라이즈` / `!목표주가변경` / `!비교 [경쟁사]`\n\n"
            "📈 **차트 및 추세 분석**\n"
            "• `!추세 [종목] [기간]` (예: `!추세 삼성전자 6개월`)\n"
            "• `!트렌드 [종목]`\n"
            "• `!손절가 [종목]`\n\n"
            "💰 **매매 판단 및 거시분석**\n"
            "• `!투자검사 [종목]`\n"
            "• `!본전 [종목]`\n"
            "• `!거시분석` / `!이벤트`",
            parse_mode="Markdown"
        )
        return

    if cmd in ["추세", "차트"]:
        query_parts = arg.split()
        target_name = query_parts[0] if query_parts else "삼성전자"
        period_str = query_parts[1] if len(query_parts) > 1 else "6개월"
        
        period_map = {"1개월": "1mo", "3개월": "3mo", "6개월": "6mo", "1년": "1y", "3년": "3y"}
        p = period_map.get(period_str, "6mo")

        ticker, name = QuickStockResolver.resolve(target_name)
        await update.message.reply_text(f"📊 [{name}] 주가 데이터 수집 및 차트 시각화 중...")
        
        try:
            df = await asyncio.to_thread(yf.download, ticker, period=p, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty:
                await update.message.reply_text("❌ 종목 데이터를 찾을 수 없습니다.")
                return

            img_bytes, cp = await asyncio.to_thread(generate_chart, df, f"{name} ({ticker})")
            report = await asyncio.to_thread(AIServiceRouter.analyze, f"종목: {name}({ticker}), 현재가: {cp:,.2f}원. 차트와 기술적 지표를 바탕으로 추세, 지지/저항, 단기·중기 전망을 분석해주세요.", img_bytes)
            
            await update.message.reply_photo(photo=img_bytes, caption=f"📊 [{name}] 기술적 추세 분석")
            await update.message.reply_text(report, parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"⚠️ 오류 발생: {e}")
        return

    if cmd in ["손절가"]:
        ticker, name = QuickStockResolver.resolve(arg)
        try:
            df = await asyncio.to_thread(yf.download, ticker, period="5d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            cp = float(df['Close'].iloc[-1])
            
            await update.message.reply_text(
                f"🛡️ **[{name}] ({ticker}) 3단계 손절 가격 산출**\n\n"
                f"• 현재 종가: `{cp:,.2f}원`\n"
                f"• 보수적 손절선 (-3%): `{cp * 0.97:,.2f}원`\n"
                f"• 중립적 손절선 (-6%): `{cp * 0.94:,.2f}원`\n"
                f"• 공격적 손절선 (-10%): `{cp * 0.90:,.2f}원`",
                parse_mode="Markdown"
            )
        except Exception as e:
            await update.message.reply_text(f"⚠️ 오류 발생: {e}")
        return

    if cmd in ["투자검사"]:
        ticker, name = QuickStockResolver.resolve(arg)
        try:
            df = await asyncio.to_thread(yf.download, ticker, period="3mo", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            cp = float(df['Close'].iloc[-1])
            
            prompt = f"종목: {name}({ticker}), 현재가: {cp}원. 수급, 뉴스, 실적, 차트, 지지/저항을 종합 검사하여 🟢 매수 적절 / 🟡 조건부 매수 / 🔴 매수 부적절 판정과 함께 매수 적정가, 손절가, 1·2차 익절가를 제시해주세요."
            report = await asyncio.to_thread(AIServiceRouter.analyze, prompt)
            await update.message.reply_text(f"🔍 **[{name}] 종합 투자검사 결과**\n\n" + report, parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"⚠️ 오류 발생: {e}")
        return

    general_queries = {
        "뉴스분석": f"다음 뉴스/링크 내용에 대해 호재/악재 여부, 단기·중기 영향, 핵심 근거와 리스크를 분석해주세요: {arg}",
        "뉴스기간": f"종목 '{arg}'에 대한 최근 기간별 뉴스를 분석하고, 사업·실적·수급·주가 연관성 및 선반영 여부를 분석해주세요.",
        "실적발표": f"종목 '{arg}'의 실적 발표 결과, 컨센서스 비교, YoY/QoQ, 향후 분기 전망을 분석해주세요.",
        "저평가": "최근 분기 영업이익 및 매출 성장, PER/PBR/ROE를 기준으로 저평가 유망 종목을 탐색하고 분석해주세요.",
        "서프라이즈": "최근 6개월간 실적 서프라이즈를 기록한 주요 종목들의 특징과 주가 반응을 분석해주세요.",
        "목표주가변경": f"종목 '{arg}'의 최근 증권사 목표주가 상향/하향 내역과 변경 이유, 현재가 괴리율을 분석해주세요.",
        "비교": f"종목 '{arg}'에 대해 경쟁사들과의 기술력, 수익성, 글로벌 점유율, 성장성, 경쟁우위 및 리스크를 비교 분석해주세요.",
        "트렌드": f"종목 '{arg}'에 대해 거래량 증가, 외국인 순매수 전환, 신규 사업 뉴스를 중심으로 추세전환 여부를 분석해주세요.",
        "본전": f"종목 '{arg}'의 현재가 기준 본전까지 필요한 상승률 및 추가매수(물타기) 타당성을 분석해주세요.",
        "거시분석": "현재 금리, 환율, 물가, 유가, 미국 금리, 경기 및 고용 지표가 주식 시장에 미치는 영향을 거시적으로 분석해주세요.",
        "이벤트": "실적 발표, 정책 발표, 신제품, 임상, 수주 등 주요 경제 이벤트가 주가에 미치는 영향과 선반영 여부를 분석해주세요.",
        "보유종목": "등록된 보유종목들의 일일 뉴스, 수급, 차트 점검 가이드라인을 제공합니다.",
        "포트폴리오": "전체 포트폴리오의 위험 분산, 업종 편중, 수익 기여도, 손실 위험 균형을 분석합니다.",
        "투자복기": "과거 실제 매매 내역(매수/매도 이유, 당시 뉴스/수급/차트, 전략적·심리적 실수) 복기 가이드를 제공합니다.",
        "투자일기": "투자 기록을 누적하여 반복되는 실수(추격매수, 손절 지연, 과도한 물타기 등)를 분석합니다."
    }

    if cmd in general_queries:
        await update.message.reply_text(f"⏳ `{cmd}` 분석을 수행 중입니다. 잠시만 기다려주세요...", parse_mode="Markdown")
        report = await asyncio.to_thread(AIServiceRouter.analyze, general_queries[cmd])
        await update.message.reply_text(report, parse_mode="Markdown")
        return

    await update.message.reply_text(f"⚠️ 알 수 없는 명령어입니다: `!{cmd}`\n도움말을 보려면 `!help`를 입력하세요.", parse_mode="Markdown")


# ==========================================
# 7. Flask 웹 서버
# ==========================================
web_app = Flask(__name__)

@web_app.route('/')
def home():
    return "Telegram Comprehensive Stock Bot with GPT-OSS-120B & Gemini 3.8 Flash is running live!", 200

def run_web():
    port = int(os.environ.get("PORT", 10000))
    web_app.run(host="0.0.0.0", port=port)


# ==========================================
# 8. 메인 실행 함수
# ==========================================
def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN이 설정되지 않았습니다!")
        return

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_all_commands))
    application.add_handler(MessageHandler(filters.COMMAND, handle_all_commands))

    threading.Thread(target=run_web, daemon=True).start()
    logger.info("🌐 Flask 웹 서버 스레드가 시작되었습니다.")

    logger.info("🤖 텔레그램 봇 폴링을 시작합니다...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
```어떤 내용(그거)을 말씀하시는지 조금만 더 알려주시면,
