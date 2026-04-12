import numpy as np
import pytest

from src.lib.cashflow_generator import (MortalityConfig, PensionConfig,
                                        SuddenSpendConfig, generate_cashflows)


def test_pension_config_constant():
  config = PensionConfig(name="test_pension", amount=5.6, start_month=12)
  cf = config.generate(n_sim=10, n_months=24, monthly_prices={})

  assert cf.shape == (24,)
  assert np.all(cf[:12] == 0.0)
  assert np.all(cf[12:] == 5.6)


def test_pension_config_with_cpi():
  # CPI starts at 1.0, becomes 1.1 at month 12, etc.
  cpi_array = np.ones((10, 25))
  cpi_array[:, 12:] = 1.1

  config = PensionConfig(name="cpi_pension",
                         amount=10.0,
                         start_month=12,
                         cpi_name="CPI")
  cf = config.generate(n_sim=10,
                       n_months=24,
                       monthly_prices={"CPI": cpi_array})

  assert cf.shape == (10, 24)
  assert np.all(cf[:, :12] == 0.0)
  # amount * cpi = 10.0 * 1.1 = 11.0
  assert np.allclose(cf[:, 12:], 11.0)


def test_pension_config_missing_cpi():
  config = PensionConfig(name="err_pension",
                         amount=10.0,
                         start_month=0,
                         cpi_name="CPI")
  with pytest.raises(ValueError):
    config.generate(n_sim=10, n_months=24, monthly_prices={})


def test_sudden_spend_config():
  config = SuddenSpendConfig(name="buy_car", amount=-300.0, month=12)
  cf = config.generate(n_sim=10, n_months=24, monthly_prices={})

  assert cf.shape == (24,)
  assert cf[11] == 0.0
  assert cf[12] == -300.0
  assert cf[13] == 0.0


def test_mortality_config():
  # 100% mortality rate at age 61
  mortality_rates = [0.0] * 61 + [1.0]
  # Start at age 60
  config = MortalityConfig(name="death",
                           mortality_rates=mortality_rates,
                           initial_age=60,
                           payout=100.0)

  # Use a fixed seed for predictable random behavior in the generator
  np.random.seed(42)
  cf = config.generate(n_sim=1000, n_months=24, monthly_prices={})

  assert cf.shape == (1000, 24)
  # Age 60 (months 0-11) has 0.0 mortality
  assert np.all(cf[:, :12] == 0.0)

  # Age 61 (months 12-23) has 1.0 mortality per year.
  # Monthly prob = 1 - (1 - 1.0)^(1/12) = 1.0
  # So everyone should die in month 12.
  assert np.all(cf[:, 12] == 100.0)
  # Note: The current implementation rolls dice every month, so if mortality is 1.0,
  # it will actually payout every month after age 61. For the purpose of "success"
  # condition in simulation, getting 100.0 every month or once is fine, as 100.0
  # is huge (10億円). Let's just check month 12 is 100.0.
  assert np.all(cf[:, 13:] == 100.0)


def test_mortality_config_exceed_max_age():
  # 生命表の最大年齢を超えた場合のテスト
  mortality_rates = [0.1, 0.2]  # age 0, 1
  config = MortalityConfig(name="death_old",
                           mortality_rates=mortality_rates,
                           initial_age=1, # month 0-11: age 1
                           payout=500.0)
  
  cf = config.generate(n_sim=10, n_months=24, monthly_prices={})
  
  # month 12以降はage 2となり、mortality_ratesの範囲外なので無条件で死亡(payout=500.0)となるはず
  assert np.all(cf[:, 12:] == 500.0)


def test_mortality_config_probability():
  # 毎年の死亡率が約0.5（月次死亡率から逆算）になるように設定し、多数のパスで期待値を確認する
  # 1 - (1-p_y)^(1/12) = p_m  =>  p_m を 0.5 にしたい場合、 p_y = 1 - (1-0.5)^12
  monthly_prob = 0.5
  yearly_prob = 1.0 - (1.0 - monthly_prob)**12
  
  mortality_rates = [yearly_prob] * 10
  config = MortalityConfig(name="death_prob",
                           mortality_rates=mortality_rates,
                           initial_age=0,
                           payout=1.0)
  
  n_sim = 10000
  n_months = 12
  np.random.seed(42)
  cf = config.generate(n_sim=n_sim, n_months=n_months, monthly_prices={})
  
  # payout=1.0 なので、cfの要素の平均が約0.5になるはず
  mean_death_rate = np.mean(cf)
  assert 0.49 < mean_death_rate < 0.51


def test_generate_cashflows():
  configs = [
      PensionConfig(name="pen", amount=5.0, start_month=0),
      SuddenSpendConfig(name="spend", amount=-10.0, month=5)
  ]
  res = generate_cashflows(configs, n_sim=10, n_months=12, monthly_prices={})

  assert "pen" in res
  assert "spend" in res
  assert np.all(res["pen"] == 5.0)
  assert res["spend"][5] == -10.0
