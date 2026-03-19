from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.lib.data_collection import fetch_asset_data, fetch_shiller_sp500


def test_fetch_shiller_sp500():
  # BeautifulSoupとrequestsのモックを作成してテスト
  with patch('src.lib.data_collection.requests.get') as mock_get:
    mock_response = MagicMock()
    # テスト用のHTMLテーブルを用意
    mock_response.content = b'''
        <html><body><table>
        <tr><th>Year</th><th>Month</th><th>Return (%)</th><th>Amount ($)</th></tr>
        <tr><td>1871</td><td>1</td><td>1.84%</td><td>101.84</td></tr>
        <tr><td>1871</td><td>2</td><td>2.93%</td><td>104.82</td></tr>
        </table></body></html>
        '''
    mock_get.return_value = mock_response

    result = fetch_shiller_sp500()

    assert isinstance(result, pd.Series)
    assert len(result) == 2
    assert result.index[0] == pd.Timestamp('1871-02-01')
    assert result.iloc[0] == 101.84


def test_fetch_asset_data():
  with patch('src.lib.data_collection.yf.download') as mock_download, \
       patch('src.lib.data_collection.fetch_shiller_sp500') as mock_shiller:

    # モックの日次データ
    dates = pd.date_range(start='2000-01-01', periods=3, freq='D')
    mock_df = pd.DataFrame(
        {
            ('Close', '^SP500TR'): [100.0, 101.0, 102.0],
            ('Close', 'ACWI'): [50.0, 51.0, 52.0],
            ('Close', 'BTC-USD'): [10.0, 11.0, 12.0]
        },
        index=dates)

    # yfinanceがMultiIndexカラムを返す場合のシミュレーション
    mock_download.return_value = mock_df

    # モックの月次データ
    shiller_dates = pd.date_range(start='1999-12-01', periods=2, freq='MS')
    mock_shiller_series = pd.Series([50.0, 52.0], index=shiller_dates)
    mock_shiller.return_value = mock_shiller_series

    result = fetch_asset_data()

    assert isinstance(result, pd.DataFrame)
    assert "Date" in result.columns
    assert "SP500" in result.columns
    assert "ACWI" in result.columns
    assert "BTC" in result.columns
    assert len(result) >= 3  # 日次データが含まれていること
