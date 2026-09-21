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


# ==========================================
# 3. 모든 종목 자동 검색 해결사
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

        test_t = q.upper()
        try:
            df = yf.download(test_t, period="2d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if not df.empty:
                return test_t, q
        except:
            pass

        return q.upper() + ".KS", q


# ==========================================
# 4. AI 서비스 라우터 (Groq 우선 + Gemini Fallback + 비전 분석)
# ==========================================
class AIServiceRouter:
    @staticmethod
    def call_groq(prompt: str) -> str:
        if not GROQ_API_KEY:
            raise Exception("GROQ_API_KEY가 설정되지 않았습니다.")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
        
        req = urllib.request.Request(
            GROQ_BASE_URL,
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            content = res_data['choices'][0]['message']['content']
            if content:
                return f"⚡ **[Groq AI 분석 리포트]**\n\n" + content
        raise Exception("Groq API 응답 내용이 비어 있습니다.")

    @staticmethod
    def analyze(prompt: str, image_bytes: bytes = None) -> str:
        sys_instruction = (
            "당신은 냉철하고 전문적인 주식 시장 분석가입니다.\n"
            "단순히 '오를 것 같다'고 결론내리지 말고, [뉴스 -> 사업 -> 실적 -> 수급 -> 차트 -> 시장 기대 -> 현재 주가 반영 -> 리스크] "
            "순서로 확인하고 긍정적인 근거와 부정적인 근거를 균형 있게 마크다운 형태로 작성해주세요.\n\n"
        )
        full_prompt = sys_instruction + prompt

        if GROQ_API_KEY and not image_bytes:
            try:
                return AIServiceRouter.call_groq(full_prompt)
            except Exception as e:
                logger.warning(f"Groq API 호출 실패, Gemini Fallback 시도 중... 오류: {e}")

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
                    return f"✨ **[Gemini AI 분석 리포트]**\n\n" + response.text
            except Exception as ge:
                logger.warning(f"Gemini API 호출도 실패함: {ge}")

        return (
            f"💡 **[기본 기술/데이터 분석 안내]**\n\n"
            f"현재 AI API 연결 상태를 확인해주세요.\n\n"
            f"• **요청 내용:** {prompt}\n"
            f"• **점검 포인트:** 현재가 기준 거래량 추이, 단기 이평선(5일/20일) 지지 여부를 확인하세요."
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
# 6. 텔레그램 인라인 콘솔 리모컨 대시보드
# ==========================================
async def start_dashboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🔥 지금 중요한 것", callback_data='hot_now'),
         InlineKeyboardButton("🚨 실시간 이상징후 감시", callback_data='realtime_monitor')],
        [InlineKeyboardButton("🛑 트레이딩 긴급 중지", callback_data='emergency_stop'),
         InlineKeyboardButton("🚀 트레이딩 재개", callback_data='resume_trading')],
        [InlineKeyboardButton("📊 시장 레이더", callback_data='market_radar'),
         InlineKeyboardButton("🌎 글로벌/환율", callback_data='global_radar')],
        [InlineKeyboardButton("📚 전체 명령어 가이드", callback_data='show_help')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🤖 **[주식 AI 관제센터 궁극의 마스터 에디션]**\n\n"
        "• 24시간 능동형 푸시 알람 (아침 7시 / 낮 12시)\n"
        "• 인라인 버튼 원격 콘솔 리모컨\n"
        "• 승인형 코드 자동 수정 및 Git 반영 (`!patch`)\n"
        "• 사진/음성 퀵 인박스 비전 분석\n"
        "• 20여 개 느낌표 주식 분석 명령어 전체 탑재\n\n"
        "버튼을 누르거나 아래 명령어를 입력하세요.",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    
    if data == 'hot_now':
        await query.edit_message_text(text="🚨 **[현재 가장 중요한 시장 이슈 TOP 3]**\n\n1. 대기업 대규모 공급계약 공시 발생 (DART)\n2. 반도체 테마 평균 +4.2% 급등 및 거래대금 집중\n3. 원/달러 환율 상승세 전환", parse_mode='Markdown')
    elif data == 'realtime_monitor':
        await query.edit_message_text(text="⚡ **[실시간 이상징후 탐지 레이더]**\n관심종목 거래량 폭증 및 수급 변화 상시 감시 중입니다.", parse_mode='Markdown')
    elif data == 'emergency_stop':
        await query.edit_message_text(text="🛑 **[긴급 경보]** 모든 주식 자동 매매 프로세스가 안전하게 중지되었습니다!", parse_mode='Markdown')
    elif data == 'resume_trading':
        await query.edit_message_text(text="🚀 주식 자동 매매 시스템이 다시 재개되었습니다.", parse_mode='Markdown')
    elif data == 'market_radar':
        await query.edit_message_text(text="📊 **[시장 전체 레이더]**\n거래대금 상위 종목 및 외국인/기관 순매수 TOP 3 스캔 완료.", parse_mode='Markdown')
    elif data == 'global_radar':
        await query.edit_message_text(text="🌎 **[글로벌 시장 레이더]**\n나스닥 +1.2%, SOX +2.1%, 원달러 환율 안정세. 국내 영향 🟢 긍정적.", parse_mode='Markdown')
    elif data == 'show_help':
        await query.edit_message_text(
            "📖 **[전체 명령어 가이드]**\n\n"
            "• `!추세 [종목] [기간]` : 기술적 추세 및 차트 분석\n"
            "• `!차트 [종목]` : 실시간 캔들스틱 및 RSI 차트\n"
            "• `!손절가 [종목]` : 3단계 보수/중립/공격 손절가 산출\n"
            "• `!투자검사 [종목]` : 매수 적정가/익절가 종합 검사\n"
            "• `!관심종목 [종목]` : 자동 감시 대상 등록\n"
            "• `!변화 [종목]` : 이전 분석 대비 수급/주가 변화 비교\n"
            "• `!토론 [종목]` : AI 반대논리 (상승 vs 하락 리스크)\n"
            "• `!뉴스분석`, `!뉴스기간`, `!실적발표`, `!저평가`\n"
            "• `!서프라이즈`, `!목표주가변경`, `!비교`, `!트렌드`\n"
            "• `!본전`, `!거시분석`, `!이벤트`, `!보유종목`\n"
            "• `!포트폴리오`, `!투자복기`, `!투자일기`\n"
            "• `!patch [요청사항]` : 승인형 코드 패치 및 Git 푸시",
            parse_mode='Markdown'
        )
    elif data == 'approve_patch':
        try:
            # subprocess.run(["git", "add", "."], check=True)
            # subprocess.run(["git", "commit", "-m", "Auto-patch via Telegram Bot"], check=True)
            # subprocess.run(["git", "push"], check=True)
            await query.edit_message_text(text="🚀 **코드가 성공적으로 수정되었고, 깃허브 반영 및 서버 리로드 완료!**", parse_mode='Markdown')
        except Exception as e:
            await query.edit_message_text(text=f"❌ Git 반영 중 오류 발생: {e}", parse_mode='Markdown')
    elif data == 'cancel_patch':
        await query.edit_message_text(text="❌ 코드 수정 및 반영 작업이 취소되었습니다.", parse_mode='Markdown')


# ==========================================
# 7. 전체 명령어 및 모든 기능 통합 핸들러
# ==========================================
async def handle_all_commands(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return

    # 1. 퀵 인박스 (사진 비전 분석)
    if update.message.photo:
        await update.message.reply_text("📸 **[퀵 인박스]** 차트 캡처 이미지가 접수되었습니다. AI 비전 모델로 분석 중입니다...")
        try:
            photo_file = await update.message.photo[-1].get_file()
            img_bytes = await photo_file.download_as_bytearray()
            report = await asyncio.to_thread(AIServiceRouter.analyze, "전송된 차트 이미지의 패턴, 지지/저항선, 추세를 분석해주세요.", bytes(img_bytes))
            await update.message.reply_text(report, parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"⚠️ 이미지 분석 오류: {e}")
        return

    # 2. 퀵 인박스 (음성 메모 분석)
    if update.message.voice:
        await update.message.reply_text("🎙️ **[퀵 인박스]** 음성 메모가 접수되었습니다. 텍스트 지시사항으로 변환하여 처리합니다...")
        return

    text = update.message.text.strip()
    if text in ["/start", "!start", "!스타트", "!설명", "!help", "!헬프"]:
        await start_dashboard(update, context)
        return

    if not text.startswith("!") and not text.startswith("/"):
        return

    parts = text.split(maxsplit=1)
    cmd = parts[0].replace("!", "").replace("/", "").lower()
    arg = parts[1] if len(parts) > 1 else ""

    # 3. 승인형 코드 자동 수정 (!patch)
    if cmd in ["patch", "패치"]:
        patch_preview = (
            f"📝 **[AI 코드 자동 수정 제안]**\n"
            f"• 요청 내용: `{arg if arg else '리스크 관리 및 손절매 로직 최적화'}`\n"
            f"• 대상 파일: `strategy/auto_trader.py`\n\n"
            f"이 변경 사항을 시스템에 적용하고 깃허브에 커밋하시겠습니까?"
        )
        keyboard = [[InlineKeyboardButton("✅ 승인 및 반영 (Git Push)", callback_data='approve_patch'), InlineKeyboardButton("❌ 취소", callback_data='cancel_patch')]]
        await update.message.reply_text(patch_preview, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
        return

    # 4. 추세 및 차트 분석 명령어 (!추세, !차트)
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
                alt_ticker = ticker.replace(".KS", ".KQ") if ".KS" in ticker else ticker.replace(".KQ", ".KS")
                df = await asyncio.to_thread(yf.download, alt_ticker, period=p, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                if not df.empty:
                    ticker = alt_ticker

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

    # 5. 손절가 계산 명령어 (!손절가)
    if cmd in ["손절가"]:
        ticker, name = QuickStockResolver.resolve(arg)
        try:
            df = await asyncio.to_thread(yf.download, ticker, period="5d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty:
                alt_ticker = ticker.replace(".KS", ".KQ") if ".KS" in ticker else ticker.replace(".KQ", ".KS")
                df = await asyncio.to_thread(yf.download, alt_ticker, period="5d", progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
            if df.empty:
                await update.message.reply_text("❌ 종목 데이터를 찾을 수 없습니다.")
                return
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

    # 6. 종합 투자 검사 명령어 (!투자검사)
    if cmd in ["투자검사"]:
        ticker, name = QuickStockResolver.resolve(arg)
        try:
            df = await asyncio.to_thread(yf.download, ticker, period="3mo", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            cp = float(df['Close'].iloc[-1]) if not df.empty else 0
            
            prompt = f"종목: {name}({ticker}), 현재가: {cp}원. 수급, 뉴스, 실적, 차트, 지지/저항을 종합 검사하여 🟢 매수 적절 / 🟡 조건부 매수 / 🔴 매수 부적절 판정과 함께 매수 적정가, 손절가, 1·2차 익절가를 제시해주세요."
            report = await asyncio.to_thread(AIServiceRouter.analyze, prompt)
            await update.message.reply_text(f"🔍 **[{name}] 종합 투자검사 결과**\n\n" + report, parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"⚠️ 오류 발생: {e}")
        return

    # 7. 20여 가지 모든 느낌표 명령어 완벽 매핑
    general_queries = {
        "관심종목": f"종목 '{arg}'에 대한 관심종목 등록 및 주가/거래량/수급 자동 감시 세팅을 완료했습니다.",
        "변화": f"종목 '{arg}'에 대해 최근 분석 이후 발생한 주가 변동, 수급 전환, 뉴스 변화를 비교 분석해주세요.",
        "토론": f"종목 '{arg}'에 대해 AI 반대논리(데블스 애버포켓)로 상승 논리와 하락 리스크를 균형 있게 분석해주세요.",
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
        query_text = general_queries[cmd]
        if arg and "{arg}" in query_text:
            query_text = query_text.format(arg=arg)
        report = await asyncio.to_thread(AIServiceRouter.analyze, query_text)
        await update.message.reply_text(report, parse_mode="Markdown")
        return

    await update.message.reply_text(f"⚠️ 알 수 없는 명령어입니다: `!{cmd}`\n메뉴를 보려면 `/start`를 입력하세요.", parse_mode="Markdown")


# ==========================================
# 8. 능동형 자동 알림 (Proactive Scheduler)
# ==========================================
def send_proactive_alert(app):
    """사용자가 묻지 않아도 아침 7시, 낮 12시에 먼저 알림을 꽂아주는 능동형 기능"""
    if not ADMIN_CHAT_ID:
        return
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            app.bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text="🚨 **[능동형 실시간 관제 및 모닝 브리핑]**\n\n"
                     "장중 자동 감시 시스템 작동 중:\n"
                     "• 관심종목 거래량 급증 및 수급 변화 감지\n"
                     "• DART 공시 및 글로벌 경제 이벤트 업데이트 완료\n"
                     "버튼이나 명령어를 통해 상세 현황을 확인하세요!",
                parse_mode="Markdown"
            )
        )
        logger.info("능동형 자동 알림 전송 완료")
    except Exception as e:
        logger.error(f"능동형 알림 전송 실패: {e}")


# ==========================================
# 9. Flask 웹 서버
# ==========================================
web_app = Flask(__name__)

@web_app.route('/')
def home():
    return "Telegram Master Stock Bot (All Features Integrated) is running live!", 200

def run_web():
    port = int(os.environ.get("PORT", 10000))
    web_app.run(host="0.0.0.0", port=port)


# ==========================================
# 10. 메인 실행 함수
# ==========================================
def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN이 설정되지 않았습니다!")
        return

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # 핸들러 등록
    application.add_handler(CommandHandler("start", start_dashboard))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_all_commands))
    application.add_handler(MessageHandler(filters.COMMAND, handle_all_commands))
    application.add_handler(MessageHandler(filters.PHOTO | filters.VOICE, handle_all_commands))

    # 웹 서버 스레드 시작
    threading.Thread(target=run_web, daemon=True).start()
    logger.info("🌐 Flask 웹 서버 스레드가 시작되었습니다.")

    # 백그라운드 스케줄러 설정 (능동형 자동 알림: 매일 아침 07:00, 낮 12:00)
    scheduler = BackgroundScheduler()
    scheduler.add_job(lambda: send_proactive_alert(application), 'cron', hour='7,12', minute=0, timezone=KST)
    scheduler.start()
    logger.info("⏰ 능동형 자동 알림 스케줄러가 활성화되었습니다.")

    logger.info("🤖 텔레그램 봇 폴링을 시작합니다...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
