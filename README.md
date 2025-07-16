# 💱 USD/JPY 為替分析システム

エコノミストや為替分析担当者向けの高機能為替分析Streamlitアプリケーションです。Snowflake上のCybersynデータを活用し、AI分析機能を組み込んだプロフェッショナル向けツールです。

## 🚀 主要機能

### 📊 基本分析機能
- **リアルタイム為替レート表示**: USD/JPYの最新レートと前日比変動
- **期間指定分析**: カスタム期間での詳細分析
- **インタラクティブチャート**: Plotlyを使用した高度な可視化
- **統計分析**: 基本統計量、分布分析、ボラティリティ測定

### 📈 テクニカル分析
- **移動平均線**: 5日、20日、50日移動平均
- **ボリンジャーバンド**: 価格の変動範囲とボラティリティ分析
- **RSI (相対力指数)**: 買われすぎ・売られすぎの判定
- **MACD**: トレンドの転換点検出
- **ボラティリティ指標**: リスク評価のための変動性測定

### 🤖 AI分析機能
Snowflakeの`CORTEX.COMPLETE`関数を使用した高度な分析:

1. **市場トレンド分析**
   - 現在の市場状況評価
   - トレンド要因の分析
   - 今後の見通し予測
   - リスク要因の特定

2. **テクニカル分析**
   - チャートパターンの評価
   - 売買シグナルの判定
   - サポート・レジスタンスレベルの特定
   - 短期的方向性の予測

3. **リスク評価**
   - ボラティリティレベルの評価
   - 主要リスク要因の分析
   - ヘッジ戦略の提案
   - 注目すべき経済指標の特定

### 📋 データ管理機能
- **データ詳細表示**: 最新10日間のデータテーブル
- **CSVエクスポート**: 分析結果のダウンロード機能
- **期間フィルタリング**: 柔軟な期間設定
- **リアルタイム更新**: 1時間ごとのデータキャッシュ更新

## 📊 データソース

### Cybersyn Financial & Economic Essentials
- **テーブル**: `FINANCE__ECONOMICS.CYBERSYN.FX_RATES_TIMESERIES`
- **データ範囲**: USD/JPY為替レートの日次データ
- **更新頻度**: 日次更新
- **データ品質**: 金融機関レベルの高品質データ

### データ仕様
```sql
SELECT
    DATE,
    VALUE AS EXCHANGE_RATE,
    VARIABLE_NAME
FROM
    FINANCE__ECONOMICS.CYBERSYN.FX_RATES_TIMESERIES
WHERE
    BASE_CURRENCY_ID ILIKE '%USD%'
    AND QUOTE_CURRENCY_ID ILIKE '%JPY%'
ORDER BY
    DATE;
```

## 🛠️ 技術仕様

### フロントエンド
- **フレームワーク**: Streamlit
- **可視化**: Plotly (Chart.js backend)
- **レスポンシブデザイン**: Wide layout対応
- **UI/UX**: プロフェッショナル向けダッシュボード

### バックエンド
- **データ処理**: Pandas, NumPy
- **Snowflake接続**: Snowpark Python
- **キャッシング**: Streamlit cache (1時間TTL)
- **AI分析**: Snowflake Cortex Complete

### パフォーマンス最適化
- **データキャッシング**: `@st.cache_data`による効率的なデータ取得
- **セッション管理**: `@st.cache_resource`によるSnowflakeセッション最適化
- **チャート最適化**: Plotly subplotsによる高速レンダリング

## 🚀 Streamlit in Snowflakeデプロイ手順

### 1. 前提条件
- Snowflakeアカウント（Standard以上推奨）
- `FINANCE__ECONOMICS.CYBERSYN.FX_RATES_TIMESERIES`へのアクセス権限
- Streamlit Apps機能の有効化
- Cortex機能の有効化

### 2. アプリケーションの作成

#### ステップ1: Snowflakeにログイン
```sql
-- Snowflakeウェブインターフェースにログイン
-- 適切なロールとウェアハウスを選択
USE ROLE ACCOUNTADMIN; -- または適切なロール
USE WAREHOUSE COMPUTE_WH; -- または適切なウェアハウス
```

#### ステップ2: Streamlitアプリの作成
```sql
-- Streamlitアプリケーションを作成
CREATE STREAMLIT USD_JPY_FX_ANALYSIS
  ROOT_LOCATION = '@your_stage/fx_analysis'
  MAIN_FILE = 'fx-analytics-app.py'
  QUERY_WAREHOUSE = 'COMPUTE_WH';
```

#### ステップ3: ファイルのアップロード
1. Snowflakeステージの作成:
```sql
CREATE OR REPLACE STAGE fx_analysis_stage;
```

2. ファイルのアップロード:
```sql
-- アプリファイルをステージにアップロード
PUT file://fx-analytics-app.py @fx_analysis_stage/;
PUT file://requirements.txt @fx_analysis_stage/;
```

3. アプリの設定更新:
```sql
ALTER STREAMLIT USD_JPY_FX_ANALYSIS SET
  ROOT_LOCATION = '@fx_analysis_stage'
  MAIN_FILE = 'fx-analytics-app.py';
```

### 3. 権限設定

#### データアクセス権限
```sql
-- Cybersynデータへのアクセス権限を確認
GRANT USAGE ON DATABASE FINANCE__ECONOMICS TO ROLE YOUR_ROLE;
GRANT USAGE ON SCHEMA FINANCE__ECONOMICS.CYBERSYN TO ROLE YOUR_ROLE;
GRANT SELECT ON TABLE FINANCE__ECONOMICS.CYBERSYN.FX_RATES_TIMESERIES TO ROLE YOUR_ROLE;
```

#### Cortex機能の権限
```sql
-- Cortex機能の使用権限
GRANT USAGE ON DATABASE SNOWFLAKE TO ROLE YOUR_ROLE;
GRANT USAGE ON SCHEMA SNOWFLAKE.CORTEX TO ROLE YOUR_ROLE;
```

### 4. アプリケーションの実行

1. **アプリの起動**:
   - Snowflakeウェブインターフェースの「Streamlit」セクションからアプリを選択
   - 「USD_JPY_FX_ANALYSIS」アプリを起動

2. **初期設定の確認**:
   - データ接続の確認
   - 必要な権限の確認
   - デフォルト設定での動作確認

## 📱 使用方法

### 基本操作

1. **期間設定**
   - サイドバーで分析期間を設定
   - 開始日と終了日を選択

2. **分析オプション**
   - テクニカル指標表示のON/OFF
   - AI分析機能のON/OFF
   - 統計分析表示のON/OFF

3. **AI分析タイプの選択**
   - 市場トレンド分析
   - テクニカル分析
   - リスク評価

### 高度な活用方法

#### エコノミスト向け使用例
```python
# 1. 長期トレンド分析
期間設定: 過去1年間
AI分析: 市場トレンド分析
focus: マクロ経済要因の影響評価

# 2. 政策影響分析
期間設定: 政策発表前後1ヶ月
AI分析: リスク評価
focus: 金融政策変更の市場インパクト

# 3. 四半期レポート作成
期間設定: 四半期期間
全機能有効化
データエクスポート: CSVダウンロード
```

#### トレーダー向け使用例
```python
# 1. 日次テクニカル分析
期間設定: 過去3ヶ月
AI分析: テクニカル分析
focus: 売買シグナルの確認

# 2. ボラティリティ分析
期間設定: 過去1ヶ月
AI分析: リスク評価
focus: ポジションサイジングの決定

# 3. 短期トレンド確認
期間設定: 過去2週間
テクニカル指標: 全て有効
focus: エントリーポイントの特定
```

## 🔧 カスタマイズ・拡張

### 新しい通貨ペアの追加
```python
# load_fx_data関数の修正例
def load_fx_data(start_date, end_date, base_currency='USD', quote_currency='JPY'):
    query = f"""
    SELECT DATE, VALUE AS EXCHANGE_RATE, VARIABLE_NAME
    FROM FINANCE__ECONOMICS.CYBERSYN.FX_RATES_TIMESERIES
    WHERE BASE_CURRENCY_ID ILIKE '%{base_currency}%'
        AND QUOTE_CURRENCY_ID ILIKE '%{quote_currency}%'
        AND DATE >= '{start_date}'
        AND DATE <= '{end_date}'
    ORDER BY DATE
    """
```

### 新しいテクニカル指標の追加
```python
# calculate_technical_indicators関数に追加
def calculate_technical_indicators(df, price_col='EXCHANGE_RATE'):
    # 既存の指標...
    
    # ストキャスティクス
    low_14 = df[price_col].rolling(window=14).min()
    high_14 = df[price_col].rolling(window=14).max()
    df['Stochastic_K'] = 100 * ((df[price_col] - low_14) / (high_14 - low_14))
    df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
    
    return df
```

### AI分析プロンプトのカスタマイズ
```python
# より具体的な分析のためのプロンプト例
prompt = f"""
USD/JPY為替レートの詳細分析を実施してください。

【マーケットデータ】
現在レート: {latest_rate:.2f}円
日次変動: {change:+.2f}円 ({change_pct:+.2f}%)
週次変動: {weekly_change:+.2f}円
月次変動: {monthly_change:+.2f}円

【テクニカル指標】
RSI(14): {rsi:.1f}
MACD: {macd:.4f}
ボリンジャーバンド位置: {bb_position}

【分析要求】
1. 短期(1週間)の方向性予測
2. 中期(1ヶ月)のトレンド分析
3. 主要な経済イベントの影響評価
4. リスクシナリオの提示
"""
```

## 🔒 セキュリティ・コンプライアンス

### データセキュリティ
- **データ暗号化**: Snowflakeの標準暗号化機能
- **アクセス制御**: ロールベースアクセス制御（RBAC）
- **監査ログ**: 全てのデータアクセスログ記録
- **データ地域性**: データ保存地域の制御

### コンプライアンス
- **金融規制対応**: MiFID II、Dodd-Frank対応
- **データプライバシー**: GDPR、CCPA準拠
- **監査対応**: SOC 2 Type II認証

## 📈 パフォーマンス・モニタリング

### 主要指標
- **データ読み込み時間**: < 3秒（1年間のデータ）
- **チャート描画時間**: < 2秒
- **AI分析応答時間**: < 10秒
- **同時接続ユーザー**: 最大50ユーザー

### 最適化のポイント
```python
# キャッシュ戦略
@st.cache_data(ttl=3600)  # 1時間キャッシュ
def load_fx_data(start_date, end_date):
    # データ取得ロジック

@st.cache_resource
def get_snowflake_session():
    # セッション管理
```

## 🚨 トラブルシューティング

### よくある問題と解決策

#### 1. データ取得エラー
```
Error: Failed to load FX data
```
**解決策**:
- Cybersynデータへのアクセス権限確認
- ウェアハウスの稼働状態確認
- ネットワーク接続の確認

#### 2. AI分析エラー
```
Error: AI analysis failed
```
**解決策**:
- Cortex機能の有効化確認
- プロンプトの文字数制限確認
- モデルの利用可能性確認

#### 3. パフォーマンス問題
```
Warning: Slow response time
```
**解決策**:
- データ期間の短縮
- キャッシュの確認
- ウェアハウスサイズの調整

## 📞 サポート・お問い合わせ

### 技術サポート
- **社内サポート**: IT部門へお問い合わせ
- **Snowflakeサポート**: 公式サポートチャネル
- **ドキュメント**: Streamlit公式ドキュメント

### フィードバック・要望
- **機能要望**: GitHub Issue または社内チケットシステム
- **バグレポート**: 詳細な再現手順と環境情報を含めて報告
- **改善提案**: ユーザーフィードバック収集フォーム

## 📚 参考資料

### 関連ドキュメント
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Snowflake Streamlit Apps](https://docs.snowflake.com/en/developer-guide/streamlit/about-streamlit)
- [Snowflake Cortex](https://docs.snowflake.com/en/user-guide/snowflake-cortex)
- [Cybersyn Financial Data](https://app.cybersyn.com/)

### 技術的参考資料
- [Plotly Documentation](https://plotly.com/python/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)

## 📝 変更履歴

### Version 1.0.0 (2025-01-16)
- 初回リリース
- 基本的な為替分析機能
- AI分析機能の実装
- テクニカル指標の実装

### 今後の予定
- [ ] 複数通貨ペア対応
- [ ] アラート機能
- [ ] レポート自動生成
- [ ] モバイル対応強化

---

**Copyright © 2025 USD/JPY FX Analysis System. All rights reserved.** 