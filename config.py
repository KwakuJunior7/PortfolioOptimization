"""Central configuration for the portfolio system."""
from datetime import date

TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL"]
BENCHMARK = "^GSPC"          # S&P 500 index
RISK_FREE_TICKER = "^IRX"    # 13-week US T-bill yield (in percent)
START_DATE = "2020-01-01"
END_DATE = date.today().isoformat()

TRADING_DAYS = 252
DEFAULT_RISK_FREE = 0.04     # used if the T-bill series can't be fetched
N_RANDOM_PORTFOLIOS = 10_000
MAX_WEIGHT = 0.40           # per-asset cap so Max-Sharpe does not collapse into one stock
TEST_SIZE = 0.2              # chronological hold-out fraction for ML models
RANDOM_STATE = 42

DATA_DIR = "data"
MODEL_DIR = "models"
REPORT_DIR = "reports"
