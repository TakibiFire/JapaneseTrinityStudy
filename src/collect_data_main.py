"""
アセットデータ（S&P500, ACWI, BTC）を収集し、CSVとして保存するメインスクリプト。

本スクリプトは以下の処理を行います：
1. S&P 500 (1871年〜): 
   - 1871年〜1988年まではin2013dollars.comから取得した月次トータルリターン指数（Shillerデータベース）を使用。
   - 1988年以降はyfinanceの日次トータルリターン指数（^SP500TR）を使用。
   - 重複期間を利用してスケール係数を調整し、これら2つのデータを一貫した1つのトータルリターン指数として統合しています。
   - ※データの取得元や前処理の意思決定（1ヶ月シフトによる誤差修正や、yfinanceのAdj Closeの仕様などに関する調査内容）に
     ついての詳細な仕様と注意点は、`src/lib/data_collection.py` のモジュール・ドキュメント（docstring）を参照してください。

2. MSCI ACWI (2008年〜) および BTC (2014年〜):
   - yfinanceから直接最新の日次終値（Close、yfinanceの仕様上、ETFの配当調整が反映された名目トータルリターン）を取得。
   - ACWIはグローバル株式市場、BTCは暗号資産として利用します。

3. データ統合と利用可能期間、クレンジング:
   - 全てのアセットの日付を結合し、1871年から現在までの単一の日次DataFrameにまとめます。
   - 【利用可能なデータ期間】
     - SP500: 1871-02-01 〜 現在
       （1987年末までは月次データを各月1日にマッピング、1988年以降は^SP500TRに基づく日次データ）
     - ACWI: 2008-03-28 〜 現在（日次データ）
     - BTC: 2014-09-17 〜 現在（日次データ）
   - 【CSV利用者が留意すべき事項（欠損値の扱い）】
     非営業日、週末、祝日、あるいはそのアセットが存在しなかった期間のデータは、一切の補間（前方穴埋めなど）を行わず、
     そのまま NaN として残しています。日次リターンなどの計算を行う際は、必要に応じて `.dropna()` や `.ffill()` などを活用し、
     目的（月次分析、日次分析など）に合わせた適切な前処理をご自身で実行してください。

4. 出力:
   - 最終的なデータを `data/asset_daily_prices.csv` に保存します。

実行方法:
  python3 src/collect_data_main.py [--output data/asset_daily_prices.csv]
"""
import argparse
import os

from lib.data_collection import fetch_asset_data


def main() -> None:
  """
  アセットデータ（S&P500, ACWI, BTC）を収集し、CSVとして保存するメインスクリプト。
  """
  parser = argparse.ArgumentParser(description="Collect historical asset data")
  parser.add_argument("--output",
                      type=str,
                      default="data/asset_daily_prices.csv",
                      help="Output CSV file path")
  args = parser.parse_args()

  df = fetch_asset_data()
  # 出力ディレクトリの存在を確認
  os.makedirs(os.path.dirname(args.output), exist_ok=True)
  df.to_csv(args.output, index=False)
  print(f"Data saved to {args.output}")
  print(df.head())
  print(df.tail())


if __name__ == "__main__":
  main()
