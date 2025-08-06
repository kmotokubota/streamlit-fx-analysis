import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col, to_date, when_matched, when_not_matched
import datetime
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')

# AI_COMPLETE関数用のLLMモデル選択肢
AI_COMPLETE_MODELS = [
    "llama4-maverick",
    "claude-3-5-sonnet", 
    "mistral-large2"
]

# ページ設定
st.set_page_config(
    page_title="Multi-Currency 為替分析システム",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Snowflakeセッション取得
@st.cache_resource(ttl=600)
def get_snowflake_session():
    return get_active_session()

session = get_snowflake_session()

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .analysis-section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .ai-insight {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 利用可能な通貨ペア取得関数
@st.cache_data(ttl=600) 
def get_available_currency_pairs():
    """利用可能な通貨ペアを取得"""
    query = """
    SELECT DISTINCT 
        BASE_CURRENCY_ID,
        QUOTE_CURRENCY_ID,
        BASE_CURRENCY_NAME,
        QUOTE_CURRENCY_NAME,
        VARIABLE_NAME
    FROM
        FINANCE__ECONOMICS.CYBERSYN.FX_RATES_TIMESERIES
    WHERE
        DATE >= CURRENT_DATE - 30  -- 過去30日以内にデータがある通貨ペアのみ
    ORDER BY
        BASE_CURRENCY_ID, QUOTE_CURRENCY_ID
    """
    
    df = session.sql(query).to_pandas()
    
    # 通貨ペアの表示名を作成
    df['PAIR_DISPLAY'] = df['BASE_CURRENCY_ID'] + '/' + df['QUOTE_CURRENCY_ID']
    df['PAIR_FULL_NAME'] = df['BASE_CURRENCY_NAME'] + ' / ' + df['QUOTE_CURRENCY_NAME']
    
    return df

# データ取得関数
@st.cache_data(ttl=600)
def load_fx_data(start_date, end_date, base_currency='USD', quote_currency='JPY'):
    """指定通貨ペアの為替データを取得"""
    query = f"""
    SELECT
        DATE,
        VALUE AS EXCHANGE_RATE,
        VARIABLE_NAME,
        BASE_CURRENCY_ID,
        QUOTE_CURRENCY_ID
    FROM
        FINANCE__ECONOMICS.CYBERSYN.FX_RATES_TIMESERIES
    WHERE
        BASE_CURRENCY_ID = '{base_currency}'
        AND QUOTE_CURRENCY_ID = '{quote_currency}'
        AND DATE >= '{start_date}'
        AND DATE <= '{end_date}'
    ORDER BY
        DATE
    """
    
    df = session.sql(query).to_pandas()
    if not df.empty:
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.sort_values('DATE')
    return df

# 複数通貨ペア対応データ取得関数
@st.cache_data(ttl=600)
def load_multiple_fx_data(start_date, end_date, currency_pairs):
    """複数通貨ペアの為替データを取得"""
    all_data = {}
    
    for pair in currency_pairs:
        base_currency, quote_currency = pair.split('/')
        df = load_fx_data(start_date, end_date, base_currency, quote_currency)
        if not df.empty:
            pair_name = f"{base_currency}/{quote_currency}"
            all_data[pair_name] = df
            
    return all_data

# テクニカル指標計算関数
def calculate_technical_indicators(df, price_col='EXCHANGE_RATE'):
    """テクニカル指標を計算"""
    df = df.copy()
    
    # 移動平均
    df['MA_5'] = df[price_col].rolling(window=5).mean()
    df['MA_20'] = df[price_col].rolling(window=20).mean()
    df['MA_50'] = df[price_col].rolling(window=50).mean()
    df['MA_200'] = df[price_col].rolling(window=200).mean()  # 長期トレンド用
    
    # ボリンジャーバンド
    df['BB_Middle'] = df[price_col].rolling(window=20).mean()
    bb_std = df[price_col].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # RSI
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ストキャスティクス
    high_14 = df[price_col].rolling(window=14).max()
    low_14 = df[price_col].rolling(window=14).min()
    df['Stoch_K'] = 100 * ((df[price_col] - low_14) / (high_14 - low_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # Williams %R
    df['Williams_R'] = -100 * ((high_14 - df[price_col]) / (high_14 - low_14))
    
    # ATR (Average True Range) - ボラティリティ測定
    df['High'] = df[price_col]  # 簡略化（実際は高値データが必要）
    df['Low'] = df[price_col]   # 簡略化（実際は安値データが必要）
    df['TR'] = df[price_col].diff().abs()
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # ADX (Average Directional Index) - トレンドの強さ
    # 簡略化版（実際は高値・安値データが必要）
    price_diff = df[price_col].diff()
    df['DM_Plus'] = price_diff.where(price_diff > 0, 0)
    df['DM_Minus'] = (-price_diff).where(price_diff < 0, 0)
    df['DI_Plus'] = 100 * (df['DM_Plus'].rolling(window=14).mean() / df['ATR'])
    df['DI_Minus'] = 100 * (df['DM_Minus'].rolling(window=14).mean() / df['ATR'])
    df['DX'] = 100 * abs(df['DI_Plus'] - df['DI_Minus']) / (df['DI_Plus'] + df['DI_Minus'])
    df['ADX'] = df['DX'].rolling(window=14).mean()
    
    # CCI (Commodity Channel Index)
    typical_price = df[price_col]  # 簡略化
    df['CCI'] = (typical_price - typical_price.rolling(window=20).mean()) / (0.015 * typical_price.rolling(window=20).std())
    
    # 一目均衡表の要素
    # 転換線 (9日間の高値・安値の平均)
    high_9 = df[price_col].rolling(window=9).max()
    low_9 = df[price_col].rolling(window=9).min()
    df['Tenkan_Sen'] = (high_9 + low_9) / 2
    
    # 基準線 (26日間の高値・安値の平均)
    high_26 = df[price_col].rolling(window=26).max()
    low_26 = df[price_col].rolling(window=26).min()
    df['Kijun_Sen'] = (high_26 + low_26) / 2
    
    # 先行スパン1
    df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(26)
    
    # 先行スパン2
    high_52 = df[price_col].rolling(window=52).max()
    low_52 = df[price_col].rolling(window=52).min()
    df['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)
    
    # 遅行スパン
    df['Chikou_Span'] = df[price_col].shift(-26)
    
    # ボラティリティ（20日間の標準偏差）
    df['Volatility'] = df[price_col].rolling(window=20).std()
    
    # 日次リターン
    df['Daily_Return'] = df[price_col].pct_change()
    
    # MACD
    exp1 = df[price_col].ewm(span=12).mean()
    exp2 = df[price_col].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    return df

# AI分析関数
def get_ai_analysis(df, analysis_type, currency_pair="USD/JPY", model="llama4-maverick"):
    """AI_COMPLETE関数を使用した分析"""
    
    # 最新データの準備
    latest_rate = df['EXCHANGE_RATE'].iloc[-1]
    prev_rate = df['EXCHANGE_RATE'].iloc[-2] if len(df) > 1 else latest_rate
    change = latest_rate - prev_rate
    change_pct = (change / prev_rate) * 100 if prev_rate != 0 else 0
    
    # 基本統計
    min_rate = df['EXCHANGE_RATE'].min()
    max_rate = df['EXCHANGE_RATE'].max()
    avg_rate = df['EXCHANGE_RATE'].mean()
    volatility = df['EXCHANGE_RATE'].std()
    
    # 最近のトレンド
    recent_data = df.tail(10)
    recent_trend = "上昇" if recent_data['EXCHANGE_RATE'].iloc[-1] > recent_data['EXCHANGE_RATE'].iloc[0] else "下降"
    
    if analysis_type == "market_trend":
        prompt = f"""
        {currency_pair}為替レートの市場分析をお願いします。
        
        現在のレート: {latest_rate:.4f}
        前日比: {change:+.4f} ({change_pct:+.2f}%)
        期間内最高値: {max_rate:.4f}
        期間内最安値: {min_rate:.4f}
        平均レート: {avg_rate:.4f}
        ボラティリティ: {volatility:.2f}
        最近のトレンド: {recent_trend}傾向
        
        プロのエコノミストとして、以下の観点から分析してください：
        1. 現在の市場状況の評価
        2. トレンドの要因分析
        3. 今後の見通し
        4. リスク要因
        """
        
    elif analysis_type == "technical_analysis":
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else None
        macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else None
        
        prompt = f"""
        {currency_pair}為替レートのテクニカル分析をお願いします。
        
        現在のレート: {latest_rate:.4f}
        RSI: {rsi:.1f} if rsi else 'N/A'
        MACD: {macd:.4f} if macd else 'N/A'
        ボラティリティ: {volatility:.4f}
        
        テクニカルアナリストとして、以下を分析してください：
        1. チャートパターンの評価
        2. 売買シグナルの状況
        3. サポート・レジスタンスレベル
        4. 短期的な方向性
        """
        
    elif analysis_type == "risk_assessment":
        prompt = f"""
        {currency_pair}為替レートのリスク評価をお願いします。
        
        現在のボラティリティ: {volatility:.4f}
        最近の最大変動幅: {max_rate - min_rate:.4f}
        日次変動率の標準偏差: {df['Daily_Return'].std()*100:.2f}%
        
        リスク管理の専門家として、以下を評価してください：
        1. 現在のボラティリティレベル
        2. 主要なリスク要因
        3. ヘッジ戦略の提案
        4. 注意すべき経済指標
        """
    
    try:
        # AI_COMPLETE関数の実行
        ai_query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            '{model}',
            '{prompt}'
        ) as analysis
        """
        
        result = session.sql(ai_query).collect()
        return result[0]['ANALYSIS'] if result else "AI分析を取得できませんでした。"
        
    except Exception as e:
        return f"AI分析でエラーが発生しました: {str(e)}"

# メインアプリケーション
def main():
    # ヘッダー
    st.markdown('<div class="main-header">💱 Multi-Currency 為替分析システム</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 利用可能な通貨ペアを取得
    try:
        with st.spinner("利用可能な通貨ペアを取得中..."):
            currency_pairs_df = get_available_currency_pairs()
    except Exception as e:
        st.error(f"通貨ペア情報の取得に失敗しました: {str(e)}")
        return
    
    # サイドバー設定
    with st.sidebar:
        st.header("📊 分析設定")
        
        # 通貨ペア選択
        st.subheader("💱 通貨ペア選択")
        
        # 分析モード選択
        analysis_mode = st.radio(
            "分析モード",
            ["単一通貨ペア", "複数通貨ペア比較"],
            help="単一通貨ペアは詳細分析、複数通貨ペアは比較分析が可能です"
        )
        
        if analysis_mode == "単一通貨ペア":
            # 単一通貨ペア選択
            selected_pair_idx = st.selectbox(
                "通貨ペアを選択",
                range(len(currency_pairs_df)),
                format_func=lambda x: f"{currency_pairs_df.iloc[x]['PAIR_DISPLAY']} ({currency_pairs_df.iloc[x]['PAIR_FULL_NAME']})",
                index=0 if len(currency_pairs_df) > 0 else None
            )
            
            if selected_pair_idx is not None:
                selected_pairs = [currency_pairs_df.iloc[selected_pair_idx]['PAIR_DISPLAY']]
                base_currency = currency_pairs_df.iloc[selected_pair_idx]['BASE_CURRENCY_ID']
                quote_currency = currency_pairs_df.iloc[selected_pair_idx]['QUOTE_CURRENCY_ID']
            else:
                selected_pairs = []
                base_currency = quote_currency = None
        else:
            # 複数通貨ペア選択
            available_pairs = [f"{row['PAIR_DISPLAY']} ({row['PAIR_FULL_NAME']})" for _, row in currency_pairs_df.iterrows()]
            selected_pair_names = st.multiselect(
                "通貨ペアを選択（最大5つ）",
                available_pairs,
                default=available_pairs[:3] if len(available_pairs) >= 3 else available_pairs,
                help="比較分析のため最大5つまで選択可能"
            )
            
            # 選択された通貨ペアを処理
            selected_pairs = []
            for pair_name in selected_pair_names[:5]:  # 最大5つに制限
                pair_display = pair_name.split(' (')[0]
                selected_pairs.append(pair_display)
        
        # 期間選択
        st.subheader("📅 期間選択")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "開始日",
                value=datetime.now() - timedelta(days=365),
                max_value=datetime.now().date()
            )
        with col2:
            end_date = st.date_input(
                "終了日",
                value=datetime.now().date(),
                max_value=datetime.now().date()
            )
        
        # テクニカル指標選択（単一通貨ペアモードのみ）
        if analysis_mode == "単一通貨ペア":
            st.subheader("📈 テクニカル指標")
            show_technical = st.checkbox("テクニカル指標表示", value=True)
            
            if show_technical:
                st.write("**表示する指標を選択:**")
                show_ma = st.checkbox("移動平均線", value=True)
                show_bb = st.checkbox("ボリンジャーバンド", value=True)
                show_rsi = st.checkbox("RSI", value=True)
                show_macd = st.checkbox("MACD", value=True)
                show_stoch = st.checkbox("ストキャスティクス", value=False)
                show_williams = st.checkbox("Williams %R", value=False)
                show_adx = st.checkbox("ADX (トレンド強度)", value=False)
                show_cci = st.checkbox("CCI", value=False)
                show_ichimoku = st.checkbox("一目均衡表", value=False)
                show_atr = st.checkbox("ATR (ボラティリティ)", value=False)
        else:
            # 複数通貨ペア比較モードの場合はテクニカル指標を無効化
            show_technical = False
            show_ma = show_bb = show_rsi = show_macd = False
            show_stoch = show_williams = show_adx = show_cci = show_ichimoku = show_atr = False
        
        # 分析オプション
        st.subheader("🔍 分析オプション")
        show_ai_analysis = st.checkbox("AI分析表示", value=True)
        
        # AI分析設定
        selected_model = AI_COMPLETE_MODELS[0]  # デフォルト値
        if show_ai_analysis:
            st.subheader("🤖 AI分析設定")
            selected_model = st.selectbox(
                "AIモデルを選択", 
                AI_COMPLETE_MODELS, 
                index=0,
                help="分析に使用するAIモデルを選択してください"
            )
            st.info("🤖 SnowflakeのCortex AI機能を使用して為替分析を生成します")
        
        show_statistics = st.checkbox("統計分析表示", value=True)
        show_correlation = st.checkbox("相関分析", value=analysis_mode == "複数通貨ペア比較")
        
        # AI分析タイプ
        if show_ai_analysis and analysis_mode == "単一通貨ペア":
            ai_analysis_type = st.selectbox(
                "AI分析タイプ",
                ["market_trend", "technical_analysis", "risk_assessment"],
                format_func=lambda x: {
                    "market_trend": "市場トレンド分析",
                    "technical_analysis": "テクニカル分析", 
                    "risk_assessment": "リスク評価"
                }[x]
            )
    
    # 通貨ペアが選択されているかチェック
    if not selected_pairs:
        st.warning("通貨ペアを選択してください。")
        return
    
    # データ読み込み
    try:
        with st.spinner("データを読み込んでいます..."):
            if analysis_mode == "単一通貨ペア":
                # 単一通貨ペアの場合
                df = load_fx_data(start_date, end_date, base_currency, quote_currency)
                if df.empty:
                    st.error("指定期間のデータが見つかりません。")
                    return
                # テクニカル指標計算
                df = calculate_technical_indicators(df)
                all_data = {selected_pairs[0]: df}
            else:
                # 複数通貨ペアの場合
                all_data = load_multiple_fx_data(start_date, end_date, selected_pairs)
                if not all_data:
                    st.error("指定期間のデータが見つかりません。")
                    return
                # 各通貨ペアにテクニカル指標を計算
                for pair_name in all_data:
                    all_data[pair_name] = calculate_technical_indicators(all_data[pair_name])
                df = list(all_data.values())[0]  # メトリクス表示用
        
        # メトリクス表示
        if analysis_mode == "単一通貨ペア":
            # 単一通貨ペアのメトリクス
            col1, col2, col3, col4 = st.columns(4)
            
            current_rate = df['EXCHANGE_RATE'].iloc[-1]
            prev_rate = df['EXCHANGE_RATE'].iloc[-2] if len(df) > 1 else current_rate
            change = current_rate - prev_rate
            change_pct = (change / prev_rate) * 100 if prev_rate != 0 else 0
            
            with col1:
                st.metric(
                    f"現在レート ({selected_pairs[0]})",
                    f"{current_rate:.4f}",
                    f"{change:+.4f} ({change_pct:+.2f}%)"
                )
            
            with col2:
                st.metric(
                    "期間最高値",
                    f"{df['EXCHANGE_RATE'].max():.4f}"
                )
            
            with col3:
                st.metric(
                    "期間最安値", 
                    f"{df['EXCHANGE_RATE'].min():.4f}"
                )
            
            with col4:
                volatility = df['Daily_Return'].std() * 100
                st.metric(
                    "ボラティリティ",
                    f"{volatility:.2f}%"
                )
        else:
            # 複数通貨ペアのメトリクス
            st.subheader("📊 通貨ペア別サマリー")
            
            metrics_data = []
            for pair_name, pair_df in all_data.items():
                if not pair_df.empty:
                    current_rate = pair_df['EXCHANGE_RATE'].iloc[-1]
                    prev_rate = pair_df['EXCHANGE_RATE'].iloc[-2] if len(pair_df) > 1 else current_rate
                    change = current_rate - prev_rate
                    change_pct = (change / prev_rate) * 100 if prev_rate != 0 else 0
                    volatility = pair_df['Daily_Return'].std() * 100
                    
                    metrics_data.append({
                        "通貨ペア": pair_name,
                        "現在レート": f"{current_rate:.4f}",
                        "前日比": f"{change:+.4f}",
                        "変動率": f"{change_pct:+.2f}%",
                        "期間最高値": f"{pair_df['EXCHANGE_RATE'].max():.4f}",
                        "期間最安値": f"{pair_df['EXCHANGE_RATE'].min():.4f}",
                        "ボラティリティ": f"{volatility:.2f}%"
                    })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
        
        # メインチャート
        if analysis_mode == "単一通貨ペア":
            st.subheader(f"📈 {selected_pairs[0]} 為替レート推移")
            
            # 詳細テクニカル分析チャート（単一通貨ペア用）
            subplot_count = 1
            subplot_titles = [f'{selected_pairs[0]} 為替レート']
            row_heights = [0.6]
            
            # テクニカル指標サブプロット1（MACD, RSI, ADX等）
            tech_indicators_1 = []
            if show_macd: tech_indicators_1.append("MACD")
            if show_rsi: tech_indicators_1.append("RSI")
            if show_adx: tech_indicators_1.append("ADX")
            
            if show_technical and tech_indicators_1:
                subplot_count += 1
                subplot_titles.append(f"📊 {' / '.join(tech_indicators_1)}")
                row_heights.append(0.2)
            
            # オシレーター系指標サブプロット（Stochastic, Williams %R, CCI）
            oscillator_indicators = []
            if show_stoch: oscillator_indicators.append("Stochastic")
            if show_williams: oscillator_indicators.append("Williams %R")
            if show_cci: oscillator_indicators.append("CCI")
            
            if show_technical and oscillator_indicators:
                subplot_count += 1
                subplot_titles.append(f"🔄 {' / '.join(oscillator_indicators)}")
                row_heights.append(0.2)
                
            # row_heightsの正規化
            total_height = sum(row_heights)
            row_heights = [h/total_height for h in row_heights]
            
            fig = make_subplots(
                rows=subplot_count, cols=1,
                subplot_titles=subplot_titles,
                vertical_spacing=0.15,
                row_heights=row_heights
            )
            
            # メイン価格チャート
            fig.add_trace(
                go.Scatter(
                    x=df['DATE'],
                    y=df['EXCHANGE_RATE'],
                    mode='lines',
                    name=selected_pairs[0],
                    line=dict(color='#1f77b4', width=2)
                ),
                row=1, col=1
            )
        else:
            st.subheader("📈 複数通貨ペア比較")
            
            # 複数通貨ペア比較チャート
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('為替レート比較 (正規化)', 'ボラティリティ比較'),
                vertical_spacing=0.2,
                row_heights=[0.7, 0.3]
            )
            
            # 色のパレット
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            # 各通貨ペアをプロット（正規化）
            for i, (pair_name, pair_df) in enumerate(all_data.items()):
                if not pair_df.empty:
                    # 正規化（初期値を100とする）
                    initial_rate = pair_df['EXCHANGE_RATE'].iloc[0]
                    normalized_rate = (pair_df['EXCHANGE_RATE'] / initial_rate) * 100
                    
                    fig.add_trace(
                        go.Scatter(
                            x=pair_df['DATE'],
                            y=normalized_rate,
                            mode='lines',
                            name=pair_name,
                            line=dict(color=colors[i % len(colors)], width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # ボラティリティ（20日移動標準偏差）
                    if 'Volatility' in pair_df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=pair_df['DATE'],
                                y=pair_df['Volatility'] * 100,
                                mode='lines',
                                name=f'{pair_name} Vol',
                                line=dict(color=colors[i % len(colors)], width=1, dash='dash'),
                                showlegend=False
                            ),
                            row=2, col=1
                        )
        
        # テクニカル指標の追加
        if show_technical and analysis_mode == "単一通貨ペア":
            # 移動平均線
            if show_ma:
                fig.add_trace(
                    go.Scatter(
                        x=df['DATE'],
                        y=df['MA_20'],
                        mode='lines',
                        name='MA20',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df['DATE'],
                        y=df['MA_50'],
                        mode='lines',
                        name='MA50',
                        line=dict(color='red', width=1)
                    ),
                    row=1, col=1
                )
                
                if 'MA_200' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['DATE'],
                            y=df['MA_200'],
                            mode='lines',
                            name='MA200',
                            line=dict(color='darkred', width=1)
                        ),
                        row=1, col=1
                    )
            
            # ボリンジャーバンド
            if show_bb:
                fig.add_trace(
                    go.Scatter(
                        x=df['DATE'],
                        y=df['BB_Upper'],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='gray', width=1, dash='dash'),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df['DATE'],
                        y=df['BB_Lower'],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)',
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # 一目均衡表
            if show_ichimoku:
                if 'Tenkan_Sen' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['DATE'],
                            y=df['Tenkan_Sen'],
                            mode='lines',
                            name='転換線',
                            line=dict(color='red', width=1)
                        ),
                        row=1, col=1
                    )
                    
                if 'Kijun_Sen' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['DATE'],
                            y=df['Kijun_Sen'],
                            mode='lines',
                            name='基準線',
                            line=dict(color='blue', width=1)
                        ),
                        row=1, col=1
                    )
                    
                # 雲の表示
                if 'Senkou_Span_A' in df.columns and 'Senkou_Span_B' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['DATE'],
                            y=df['Senkou_Span_A'],
                            mode='lines',
                            name='先行スパン1',
                            line=dict(color='green', width=1, dash='dot'),
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df['DATE'],
                            y=df['Senkou_Span_B'],
                            mode='lines',
                            name='先行スパン2',
                            line=dict(color='red', width=1, dash='dot'),
                            fill='tonexty',
                            fillcolor='rgba(0,255,0,0.1)',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
            
            # 第2サブプロット（テクニカル指標1）
            current_row = 2
            if subplot_count >= 2:
                # MACD
                if show_macd:
                    fig.add_trace(
                        go.Scatter(
                            x=df['DATE'],
                            y=df['MACD'],
                            mode='lines',
                            name='MACD',
                            line=dict(color='blue', width=1)
                        ),
                        row=current_row, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df['DATE'],
                            y=df['MACD_Signal'],
                            mode='lines',
                            name='Signal',
                            line=dict(color='red', width=1)
                        ),
                        row=current_row, col=1
                    )
                    
                    # MACD Histogram
                    fig.add_trace(
                        go.Bar(
                            x=df['DATE'],
                            y=df['MACD_Histogram'],
                            name='MACD Hist',
                            marker_color='green',
                            opacity=0.6
                        ),
                        row=current_row, col=1
                    )
                
                # RSI
                if show_rsi:
                    fig.add_trace(
                        go.Scatter(
                            x=df['DATE'],
                            y=df['RSI'],
                            mode='lines',
                            name='RSI',
                            line=dict(color='purple', width=2)
                        ),
                        row=current_row, col=1
                    )
                    
                    # RSI overbought/oversold lines
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
                
                # ADX
                if show_adx and 'ADX' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['DATE'],
                            y=df['ADX'],
                            mode='lines',
                            name='ADX',
                            line=dict(color='orange', width=2)
                        ),
                        row=current_row, col=1
                    )
            
            # 第3サブプロット（オシレーター系指標）
            if subplot_count >= 3:
                current_row = 3
                
                # ストキャスティクス
                if show_stoch and 'Stoch_K' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['DATE'],
                            y=df['Stoch_K'],
                            mode='lines',
                            name='Stoch %K',
                            line=dict(color='blue', width=1)
                        ),
                        row=current_row, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df['DATE'],
                            y=df['Stoch_D'],
                            mode='lines',
                            name='Stoch %D',
                            line=dict(color='red', width=1)
                        ),
                        row=current_row, col=1
                    )
                    
                    # オーバーボート/オーバーソールドライン
                    fig.add_hline(y=80, line_dash="dash", line_color="red", row=current_row, col=1)
                    fig.add_hline(y=20, line_dash="dash", line_color="green", row=current_row, col=1)
                
                # Williams %R
                if show_williams and 'Williams_R' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['DATE'],
                            y=df['Williams_R'],
                            mode='lines',
                            name='Williams %R',
                            line=dict(color='darkblue', width=1)
                        ),
                        row=current_row, col=1
                    )
                    
                    fig.add_hline(y=-20, line_dash="dash", line_color="red", row=current_row, col=1)
                    fig.add_hline(y=-80, line_dash="dash", line_color="green", row=current_row, col=1)
                
                # CCI
                if show_cci and 'CCI' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['DATE'],
                            y=df['CCI'],
                            mode='lines',
                            name='CCI',
                            line=dict(color='darkgreen', width=1)
                        ),
                        row=current_row, col=1
                    )
                    
                    fig.add_hline(y=100, line_dash="dash", line_color="red", row=current_row, col=1)
                    fig.add_hline(y=-100, line_dash="dash", line_color="green", row=current_row, col=1)
        
        # チャートレイアウト更新
        if analysis_mode == "単一通貨ペア":
            fig.update_layout(
                height=800,
                title_text=f"{selected_pairs[0]} 為替分析チャート",
                showlegend=True,
                hovermode='x unified'
            )
            
            fig.update_xaxes(title_text="日付", row=subplot_count, col=1)
            fig.update_yaxes(title_text="レート", row=1, col=1)
            
            if subplot_count >= 2:
                fig.update_yaxes(title_text="指標値", row=2, col=1)
                if show_rsi:
                    fig.update_yaxes(range=[0, 100], row=2, col=1)
            
            if subplot_count >= 3:
                fig.update_yaxes(title_text="オシレーター", row=3, col=1)
                if show_stoch:
                    fig.update_yaxes(range=[0, 100], row=3, col=1)
                elif show_williams:
                    fig.update_yaxes(range=[-100, 0], row=3, col=1)
        else:
            fig.update_layout(
                height=600,
                title_text="複数通貨ペア比較チャート",
                showlegend=True,
                hovermode='x unified'
            )
            
            fig.update_xaxes(title_text="日付", row=2, col=1)
            fig.update_yaxes(title_text="正規化レート (初期値=100)", row=1, col=1)
            fig.update_yaxes(title_text="ボラティリティ (%)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 統計分析セクション
        if show_statistics:
            st.subheader("📊 統計分析")
            
            if analysis_mode == "単一通貨ペア":
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.container(border=True):
                        st.write(f"**{selected_pairs[0]} 基本統計量**")
                        stats_df = pd.DataFrame({
                            '統計量': ['平均', '中央値', '標準偏差', '最小値', '最大値', '歪度', '尖度'],
                            '値': [
                                f"{df['EXCHANGE_RATE'].mean():.4f}",
                                f"{df['EXCHANGE_RATE'].median():.4f}",
                                f"{df['EXCHANGE_RATE'].std():.4f}",
                                f"{df['EXCHANGE_RATE'].min():.4f}",
                                f"{df['EXCHANGE_RATE'].max():.4f}",
                                f"{df['EXCHANGE_RATE'].skew():.2f}",
                                f"{df['EXCHANGE_RATE'].kurtosis():.2f}"
                            ]
                        })
                        st.dataframe(stats_df, use_container_width=True)
                
                with col2:
                    with st.container(border=True):
                        st.write("**ボラティリティ分析**")
                        
                        # ボラティリティのヒストグラム
                        fig_vol = px.histogram(
                            df.dropna(),
                            x='Daily_Return',
                            nbins=50,
                            title='日次リターンの分布',
                            labels={'Daily_Return': '日次リターン', 'count': '頻度'}
                        )
                        fig_vol.update_layout(height=300)
                        st.plotly_chart(fig_vol, use_container_width=True)
                
                # テクニカル指標サマリー
                if show_technical:
                    with st.container(border=True):
                        st.write("**現在のテクニカル指標**")
                        
                        tech_data = []
                        latest_data = df.iloc[-1]
                        
                        if show_rsi and 'RSI' in df.columns:
                            rsi_signal = "買われすぎ" if latest_data['RSI'] > 70 else "売られすぎ" if latest_data['RSI'] < 30 else "中立"
                            tech_data.append(["RSI", f"{latest_data['RSI']:.1f}", rsi_signal])
                        
                        if show_stoch and 'Stoch_K' in df.columns:
                            stoch_signal = "買われすぎ" if latest_data['Stoch_K'] > 80 else "売られすぎ" if latest_data['Stoch_K'] < 20 else "中立"
                            tech_data.append(["Stochastic %K", f"{latest_data['Stoch_K']:.1f}", stoch_signal])
                        
                        if show_adx and 'ADX' in df.columns:
                            adx_signal = "強いトレンド" if latest_data['ADX'] > 25 else "弱いトレンド" if latest_data['ADX'] > 20 else "レンジ相場"
                            tech_data.append(["ADX", f"{latest_data['ADX']:.1f}", adx_signal])
                        
                        if show_atr and 'ATR' in df.columns:
                            tech_data.append(["ATR", f"{latest_data['ATR']:.4f}", "ボラティリティ指標"])
                        
                        if tech_data:
                            tech_df = pd.DataFrame(tech_data, columns=["指標", "値", "シグナル"])
                            st.dataframe(tech_df, use_container_width=True)
            
            else:
                # 複数通貨ペアの統計分析
                with st.container(border=True):
                    st.write("**通貨ペア別統計分析**")
                    
                    stats_data = []
                    for pair_name, pair_df in all_data.items():
                        if not pair_df.empty:
                            stats_data.append({
                                "通貨ペア": pair_name,
                                "平均": f"{pair_df['EXCHANGE_RATE'].mean():.4f}",
                                "標準偏差": f"{pair_df['EXCHANGE_RATE'].std():.4f}",
                                "最小値": f"{pair_df['EXCHANGE_RATE'].min():.4f}",
                                "最大値": f"{pair_df['EXCHANGE_RATE'].max():.4f}",
                                "変動係数": f"{(pair_df['EXCHANGE_RATE'].std() / pair_df['EXCHANGE_RATE'].mean()):.4f}",
                                "歪度": f"{pair_df['EXCHANGE_RATE'].skew():.2f}"
                            })
                    
                    if stats_data:
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True)
        
        # 相関分析セクション
        if show_correlation and analysis_mode == "複数通貨ペア比較" and len(all_data) > 1:
            st.subheader("🔗 相関分析")
            
            # 相関行列の作成
            correlation_data = {}
            for pair_name, pair_df in all_data.items():
                if not pair_df.empty:
                    correlation_data[pair_name] = pair_df.set_index('DATE')['EXCHANGE_RATE']
            
            if len(correlation_data) > 1:
                corr_df = pd.DataFrame(correlation_data)
                corr_matrix = corr_df.corr()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.container(border=True):
                        st.write("**相関係数行列**")
                        st.dataframe(corr_matrix.round(3), use_container_width=True)
                
                with col2:
                    with st.container(border=True):
                        st.write("**相関ヒートマップ**")
                    
                    fig_corr = px.imshow(
                        corr_matrix,
                        title="通貨ペア相関ヒートマップ",
                        color_continuous_scale="RdBu",
                        aspect="auto"
                    )
                    fig_corr.update_layout(height=400)
                    st.plotly_chart(fig_corr, use_container_width=True)
        
        # AI分析セクション
        if show_ai_analysis:
            st.subheader("🤖 AI分析")
            
            if analysis_mode == "単一通貨ペア":
                with st.spinner("AI分析を実行中..."):
                    ai_result = get_ai_analysis(df, ai_analysis_type, selected_pairs[0], selected_model)
                
                with st.container(border=True):
                    st.write(f"**{selected_pairs[0]} AI分析結果**")
                    # AI分析結果の表示（Markdown記号を削除して読みやすく表示）
                    if ai_result:
                        # Markdown見出し記号を削除して、適切にフォーマット
                        formatted_result = ai_result.replace("#### ", "").replace("### ", "").replace("## ", "").replace("# ", "")
                        # 短いサブタイトルのみを太字に変換（50文字以内の行のみ対象）
                        formatted_result = re.sub(r'^(\d+\.\s+.{1,50})$', r'**\1**', formatted_result, flags=re.MULTILINE)
                        st.markdown(formatted_result)
            else:
                # 複数通貨ペアの場合は簡単なサマリー
                with st.container(border=True):
                    st.write("**複数通貨ペア分析サマリー**")
                    
                    summary_text = f"""**分析期間**: {start_date} ～ {end_date}
**分析対象**: {len(all_data)}通貨ペア ({', '.join(all_data.keys())})

**主要な観察点**:"""
                    
                    # 最もボラティリティの高い・低い通貨ペアを特定
                    volatilities = {}
                    for pair_name, pair_df in all_data.items():
                        if not pair_df.empty and 'Daily_Return' in pair_df.columns:
                            vol = pair_df['Daily_Return'].std() * 100
                            volatilities[pair_name] = vol
                    
                    if volatilities:
                        max_vol_pair = max(volatilities, key=volatilities.get)
                        min_vol_pair = min(volatilities, key=volatilities.get)
                        
                        summary_text += f"""

• 最高ボラティリティ: **{max_vol_pair}** ({volatilities[max_vol_pair]:.2f}%)
• 最低ボラティリティ: **{min_vol_pair}** ({volatilities[min_vol_pair]:.2f}%)

**リスク管理の観点**:
- 高ボラティリティ通貨ペアでは、より厳格なリスク管理が必要
- 低ボラティリティ通貨ペアは相対的に安定した投資対象

**ポートフォリオ多様化**:
- 複数通貨ペアの相関関係を確認し、分散効果を活用
- 相関の低いペアを組み合わせることでリスク軽減が可能"""
                    
                    st.markdown(summary_text)
        
        # データ詳細
        with st.expander("📋 データ詳細"):
            if analysis_mode == "単一通貨ペア":
                # 単一通貨ペアの詳細データ
                st.write(f"**{selected_pairs[0]} 最新のデータ (直近10日)**")
                
                # 表示する列を動的に選択
                display_columns = ['DATE', 'EXCHANGE_RATE', 'Daily_Return']
                if show_ma:
                    display_columns.extend(['MA_20', 'MA_50'])
                if show_rsi and 'RSI' in df.columns:
                    display_columns.append('RSI')
                if show_stoch and 'Stoch_K' in df.columns:
                    display_columns.extend(['Stoch_K', 'Stoch_D'])
                if show_adx and 'ADX' in df.columns:
                    display_columns.append('ADX')
                
                # 利用可能な列のみ選択
                available_columns = [col for col in display_columns if col in df.columns]
                latest_data = df[available_columns].tail(10).copy()
                
                # フォーマット
                if 'Daily_Return' in latest_data.columns:
                    latest_data['Daily_Return'] = latest_data['Daily_Return'].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
                if 'EXCHANGE_RATE' in latest_data.columns:
                    latest_data['EXCHANGE_RATE'] = latest_data['EXCHANGE_RATE'].map(lambda x: f"{x:.4f}")
                
                # 数値列のフォーマット
                for col in ['MA_20', 'MA_50', 'RSI', 'Stoch_K', 'Stoch_D', 'ADX']:
                    if col in latest_data.columns:
                        if col in ['RSI', 'Stoch_K', 'Stoch_D', 'ADX']:
                            latest_data[col] = latest_data[col].map(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
                        else:
                            latest_data[col] = latest_data[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                
                st.dataframe(latest_data, use_container_width=True)
                
                # データのダウンロード
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 データをCSVでダウンロード",
                    data=csv,
                    file_name=f"{selected_pairs[0].replace('/', '_')}_analysis_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )
            else:
                # 複数通貨ペアの詳細データ
                st.write("**複数通貨ペア 最新データ比較 (直近5日)**")
                
                # 通貨ペア別にタブ表示
                if len(all_data) > 1:
                    tabs = st.tabs(list(all_data.keys()))
                    
                    for i, (pair_name, pair_df) in enumerate(all_data.items()):
                        with tabs[i]:
                            if not pair_df.empty:
                                latest_data = pair_df[['DATE', 'EXCHANGE_RATE', 'Daily_Return']].tail(5).copy()
                                latest_data['Daily_Return'] = latest_data['Daily_Return'].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
                                latest_data['EXCHANGE_RATE'] = latest_data['EXCHANGE_RATE'].map(lambda x: f"{x:.4f}")
                                st.dataframe(latest_data, use_container_width=True)
                
                # 全データの一括ダウンロード
                if st.button("📥 全通貨ペアデータを一括ダウンロード"):
                    combined_data = []
                    for pair_name, pair_df in all_data.items():
                        pair_df_copy = pair_df.copy()
                        pair_df_copy['CURRENCY_PAIR'] = pair_name
                        combined_data.append(pair_df_copy)
                    
                    if combined_data:
                        combined_df = pd.concat(combined_data, ignore_index=True)
                        csv = combined_df.to_csv(index=False)
                        st.download_button(
                            label="💾 結合データをダウンロード",
                            data=csv,
                            file_name=f"multi_currency_analysis_{start_date}_{end_date}.csv",
                            mime="text/csv"
                        )
        
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        st.error("Snowflakeへの接続またはデータ取得に問題があります。")

# フッター
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        <p>💱 Multi-Currency 為替分析システム | Powered by Streamlit & Snowflake | 
        Data Source: Cybersyn Financial & Economic Essentials</p>
        <p style='font-size: 0.8rem;'>
        🔧 新機能: 複数通貨ペア比較 | 📈 高度テクニカル分析 | 🤖 AI市場洞察 | 🔗 相関分析
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main() 
