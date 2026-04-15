# 50歳の取り崩し最適戦略では何%ルールなのか

## 50歳から取り崩しを開始し95歳まで破綻しない確率を最大化する

<!--
DO NOT DELETE.

NOW: Read @all_60yr.md.

Think about what kind of parameters to test thoroughly.

50歳の2人世帯の平均消費出費は35万/月、非消費支出は14.5万/月。

年金: See @pension.md
60歳受給(10万/月くらい)と65歳受給(13.2万/月くらい)の両方を試す。
50~60歳まで無職の人として年金保険は払う。いくらか覚えていない

@src/pension_grid_main.py では年金の支払いを cashflow でやっていた。これが本当に妥当か・現在の挙動が正しいか疑問。
* DynamicSpending ありの時
  * DynamicSpending の年の出費って cashflow を考えてない気がする (See core.py)
  * DynamicRebalance の年支出を計算する時も考慮してるか分からない (See core.py)
* DynamicSpending なしの時 (日本人平均の支出トレンドに添わせる場合)
  * src/retired_spending_comp_main.py の非消費支出のトレンドが年金の支払いが終わる部分をすでに考慮してしまっているが、spline shape のため緩やかに下っている。本当は60歳で急に減らしたい
  * そもそも cashflow でやるべきか、annual cost でやるべきかわからない。

SIDE FIRE:
考慮するかどうかは結果をみて決める。その時に discuss する

他のパラメータで60歳では考えなくていいけど50歳で考えるべきことを教えて。

シミュレーションの共通条件と可変条件を以下に書くのが goal.
-->
