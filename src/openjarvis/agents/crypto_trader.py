"""CryptoTraderAgent — profit sweeper for EVM wallet balances.

This agent is designed for managed-agent ticks. It does not execute market
orders. Instead, it evaluates realized USDC balance vs configured capital and
optionally sweeps profit to a Binance deposit address using the wallet module.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import URLError
from urllib.request import urlopen

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent
from openjarvis.core.registry import AgentRegistry


logger = logging.getLogger(__name__)


# A managed-agent schedule and a manual API run can overlap.  Serializing the
# complete decision/execution cycle prevents both ticks from spending the same
# balance or closing the same position twice within one server process.
_TRADING_TICK_LOCK = threading.Lock()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_decimal(name: str, default: str) -> Decimal:
    raw = os.environ.get(name, default).strip()
    try:
        return Decimal(raw)
    except (InvalidOperation, ValueError):
        return Decimal(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _default_state() -> Dict[str, Any]:
    return {
        "last_sweep_at": 0,
        "last_tx_hash": "",
        "day": "",
        "sweeps_today": 0,
        "paper_last_price": "0",
        "paper_position": None,
        "paper_realized_pnl_usdc": "0",
        "paper_trades": [],
        "live_last_price": "0",
        "live_position": None,
        "live_realized_pnl_usdc": "0",
        "live_trades": [],
        "live_last_prices": {},
        "live_positions": {},
        "live_realized_pnl_usdc_by_symbol": {},
        "live_entry_momentum_pct_by_symbol": {},
        "live_exit_momentum_pct_by_symbol": {},
        "live_hold_streak_by_symbol": {},
        "live_last_action_ts_by_symbol": {},
        "live_price_history_by_symbol": {},
        "live_risk_day": "",
        "live_risk_day_start_equity_usdc": "0",
        "live_risk_day_peak_equity_usdc": "0",
        "live_risk_stop_until": 0,
        "live_consecutive_losses": 0,
        "live_risk_profiles": {},
    }


def _live_risk_scope(execution_backend: str, wallet_network: str = "") -> str:
    """Return a stable risk namespace for one execution account.

    Wallet balances on different chains and balances held by a perpetuals
    venue are not interchangeable.  Keeping their drawdown anchors together
    can create false brakes (or hide a real drawdown) when two managed agents
    share the same state file.
    """
    backend = str(execution_backend or "wallet").strip().lower() or "wallet"
    if backend == "wallet":
        network = _normalize_wallet_network(str(wallet_network or "")) or "default"
        return f"wallet:{network}"
    return backend


def _adaptive_opportunity_floor(
    *,
    base_score: Decimal,
    hold_streak: int,
    relax_after_holds: int,
    relax_step: Decimal,
    min_actions_per_hour: int,
) -> Decimal:
    """Relax entry quality only when an activity target was explicitly set."""
    if min_actions_per_hour <= 0 or hold_streak < relax_after_holds:
        return base_score
    relax_steps = hold_streak - relax_after_holds + 1
    return max(Decimal("-1"), base_score - relax_step * Decimal(relax_steps))


def _micro_edge_relax_steps(
    *,
    hold_streak: int,
    relax_after_holds: int,
    min_actions_per_hour: int,
    last_action_ts: int,
    now_ts: int,
) -> int:
    """Return zero unless the operator explicitly requested trading activity."""
    if min_actions_per_hour <= 0:
        return 0
    steps = max(0, hold_streak - relax_after_holds + 1)
    target_gap = max(300, int(3600 / max(1, min_actions_per_hour)))
    since_last_action = (
        now_ts - last_action_ts if last_action_ts > 0 else target_gap + 1
    )
    if since_last_action > target_gap:
        extra_steps = ((since_last_action - target_gap) // target_gap) + 1
        steps += min(3, int(extra_steps))
    return steps


def _state_path() -> Path:
    explicit = os.environ.get("OPENJARVIS_CRYPTO_TRADER_STATE_PATH", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return (Path.home() / ".openjarvis" / "crypto_trader_state.json").expanduser()


@dataclass
class TraderConfig:
    binance_address: str
    autosweep_enabled: bool
    base_capital_usdc: Decimal
    reserve_usdc: Decimal
    min_sweep_usdc: Decimal
    sweep_cooldown_sec: int
    max_sweeps_per_day: int
    confirm_code: str
    use_ai: bool
    ai_min_confidence: float
    default_payout_ratio: Decimal
    ai_max_payout_ratio: Decimal
    strategy_mode: str
    paper_symbol: str
    paper_notional_usdc: Decimal
    paper_take_profit_pct: Decimal
    paper_stop_loss_pct: Decimal
    paper_allow_real_sweep: bool
    live_symbol: str
    live_symbols: str
    live_execution_backend: str
    live_wallet_fallback_backend: str
    live_perps_leverage: int
    live_binance_test_order: bool
    live_wallet_network: str
    live_wallet_networks: str
    live_wallet_watch_symbols: str
    live_notional_usdc: Decimal
    growth_enabled: bool
    growth_reinvest_ratio: Decimal
    growth_max_notional_usdc: Decimal
    live_take_profit_pct: Decimal
    live_stop_loss_pct: Decimal
    live_max_position_minutes: int
    live_timeout_exit_pnl_pct: Decimal
    live_min_trade_usdc: Decimal
    live_slippage_pct: float
    live_max_actions_per_tick: int
    live_max_open_positions: int
    live_entry_momentum_pct: Decimal
    live_exit_momentum_pct: Decimal
    live_learning_enabled: bool
    live_learning_hold_streak: int
    live_learning_step_pct: Decimal
    live_learning_min_momentum_pct: Decimal
    live_learning_max_momentum_pct: Decimal
    live_min_actions_per_hour: int
    live_activity_boost_step_pct: Decimal
    live_score_momentum_weight: Decimal
    live_score_news_weight: Decimal
    live_score_regime_weight: Decimal
    live_score_position_bonus: Decimal
    live_min_opportunity_score: Decimal
    live_opportunity_relax_after_holds: int
    live_opportunity_relax_step_pct: Decimal
    live_risk_daily_max_drawdown_pct: Decimal
    live_risk_max_consecutive_losses: int
    live_risk_cooldown_minutes: int
    live_micro_tx_fee_ratio: Decimal
    live_micro_gas_buffer_usdc: Decimal
    live_micro_min_net_edge_pct: Decimal
    live_micro_relax_after_holds: int
    live_micro_relax_step_pct: Decimal
    live_micro_min_floor_pct: Decimal
    news_filter_enabled: bool
    news_feed_url: str
    news_max_items: int
    news_lookback_hours: int
    news_block_buy_score: int
    news_force_exit_score: int
    min_gas_eth: Decimal
    gas_warning_eth: Decimal
    gas_refill_enabled: bool
    gas_refill_target_eth: Decimal
    gas_refill_usdc_budget: Decimal
    ai_ollama_host: str  # e.g. "http://192.168.1.9:11434"
    ai_ollama_model: str  # e.g. "agent-default:latest"


def _load_cfg() -> TraderConfig:
    cooldown_raw = os.environ.get("OPENJARVIS_CRYPTO_TRADER_SWEEP_COOLDOWN_SEC", "900")
    max_daily_raw = os.environ.get("OPENJARVIS_CRYPTO_TRADER_MAX_SWEEPS_PER_DAY", "4")

    try:
        cooldown = max(0, int(cooldown_raw))
    except ValueError:
        cooldown = 900

    try:
        max_daily = max(1, int(max_daily_raw))
    except ValueError:
        max_daily = 4

    return TraderConfig(
        binance_address=os.environ.get(
            "OPENJARVIS_CRYPTO_TRADER_BINANCE_USDC_ADDRESS", ""
        ).strip(),
        autosweep_enabled=_env_bool(
            "OPENJARVIS_CRYPTO_TRADER_AUTOSWEEP_ENABLED", False
        ),
        base_capital_usdc=_env_decimal(
            "OPENJARVIS_CRYPTO_TRADER_BASE_CAPITAL_USDC", "0"
        ),
        reserve_usdc=_env_decimal("OPENJARVIS_CRYPTO_TRADER_KEEP_RESERVE_USDC", "25"),
        min_sweep_usdc=_env_decimal("OPENJARVIS_CRYPTO_TRADER_MIN_SWEEP_USDC", "10"),
        sweep_cooldown_sec=cooldown,
        max_sweeps_per_day=max_daily,
        confirm_code=os.environ.get(
            "OPENJARVIS_CRYPTO_TRADER_CONFIRM_CODE", ""
        ).strip(),
        use_ai=_env_bool("OPENJARVIS_CRYPTO_TRADER_USE_AI", False),
        ai_min_confidence=max(
            0.0,
            min(1.0, _env_float("OPENJARVIS_CRYPTO_TRADER_AI_MIN_CONFIDENCE", 0.65)),
        ),
        default_payout_ratio=max(
            Decimal("0"),
            min(
                Decimal("1"),
                _env_decimal("OPENJARVIS_CRYPTO_TRADER_DEFAULT_PAYOUT_RATIO", "1"),
            ),
        ),
        ai_max_payout_ratio=max(
            Decimal("0"),
            min(
                Decimal("1"),
                _env_decimal("OPENJARVIS_CRYPTO_TRADER_AI_MAX_PAYOUT_RATIO", "1"),
            ),
        ),
        strategy_mode=os.environ.get(
            "OPENJARVIS_CRYPTO_TRADER_STRATEGY_MODE", "wallet_only"
        )
        .strip()
        .lower(),
        paper_symbol=os.environ.get("OPENJARVIS_CRYPTO_TRADER_PAPER_SYMBOL", "BTCUSDT")
        .strip()
        .upper(),
        paper_notional_usdc=max(
            Decimal("0"),
            _env_decimal("OPENJARVIS_CRYPTO_TRADER_PAPER_NOTIONAL_USDC", "100"),
        ),
        paper_take_profit_pct=max(
            Decimal("0.001"),
            _env_decimal("OPENJARVIS_CRYPTO_TRADER_PAPER_TAKE_PROFIT_PCT", "0.02"),
        ),
        paper_stop_loss_pct=max(
            Decimal("0.001"),
            _env_decimal("OPENJARVIS_CRYPTO_TRADER_PAPER_STOP_LOSS_PCT", "0.01"),
        ),
        paper_allow_real_sweep=_env_bool(
            "OPENJARVIS_CRYPTO_TRADER_PAPER_ALLOW_REAL_SWEEP", False
        ),
        live_symbol=os.environ.get("OPENJARVIS_CRYPTO_TRADER_LIVE_SYMBOL", "ETHUSDT")
        .strip()
        .upper(),
        live_symbols=os.environ.get("OPENJARVIS_CRYPTO_TRADER_LIVE_SYMBOLS", "")
        .strip()
        .upper(),
        live_execution_backend=(
            os.environ.get("OPENJARVIS_CRYPTO_TRADER_LIVE_EXECUTION_BACKEND", "wallet")
            .strip()
            .lower()
        ),
        live_wallet_fallback_backend=(
            os.environ.get(
                "OPENJARVIS_CRYPTO_TRADER_LIVE_WALLET_FALLBACK_BACKEND", "none"
            )
            .strip()
            .lower()
        ),
        live_perps_leverage=max(
            1,
            min(
                100,
                int(
                    _env_float(
                        "OPENJARVIS_CRYPTO_TRADER_LIVE_PERPS_LEVERAGE",
                        1,
                    )
                ),
            ),
        ),
        live_binance_test_order=_env_bool(
            "OPENJARVIS_CRYPTO_TRADER_LIVE_BINANCE_TEST_ORDER", False
        ),
        live_wallet_network=os.environ.get(
            "OPENJARVIS_CRYPTO_TRADER_LIVE_WALLET_NETWORK", ""
        )
        .strip()
        .lower(),
        live_wallet_networks=os.environ.get(
            "OPENJARVIS_CRYPTO_TRADER_LIVE_WALLET_NETWORKS", ""
        )
        .strip()
        .lower(),
        live_wallet_watch_symbols=os.environ.get(
            "OPENJARVIS_CRYPTO_TRADER_LIVE_WALLET_WATCH_SYMBOLS", ""
        )
        .strip()
        .upper(),
        live_notional_usdc=max(
            Decimal("0"),
            _env_decimal("OPENJARVIS_CRYPTO_TRADER_LIVE_NOTIONAL_USDC", "25"),
        ),
        growth_enabled=_env_bool("OPENJARVIS_CRYPTO_TRADER_GROWTH_ENABLED", True),
        growth_reinvest_ratio=max(
            Decimal("0"),
            min(
                Decimal("1"),
                _env_decimal("OPENJARVIS_CRYPTO_TRADER_GROWTH_REINVEST_RATIO", "1"),
            ),
        ),
        growth_max_notional_usdc=max(
            Decimal("0"),
            _env_decimal("OPENJARVIS_CRYPTO_TRADER_GROWTH_MAX_NOTIONAL_USDC", "250"),
        ),
        live_take_profit_pct=max(
            Decimal("0.001"),
            _env_decimal("OPENJARVIS_CRYPTO_TRADER_LIVE_TAKE_PROFIT_PCT", "0.02"),
        ),
        live_stop_loss_pct=max(
            Decimal("0.001"),
            _env_decimal("OPENJARVIS_CRYPTO_TRADER_LIVE_STOP_LOSS_PCT", "0.01"),
        ),
        live_max_position_minutes=max(
            1,
            int(
                _env_float(
                    "OPENJARVIS_CRYPTO_TRADER_LIVE_MAX_POSITION_MINUTES",
                    240,
                )
            ),
        ),
        live_timeout_exit_pnl_pct=_env_decimal(
            "OPENJARVIS_CRYPTO_TRADER_LIVE_TIMEOUT_EXIT_PNL_PCT", "0.001"
        ),
        live_min_trade_usdc=max(
            Decimal("0"),
            _env_decimal("OPENJARVIS_CRYPTO_TRADER_LIVE_MIN_TRADE_USDC", "5"),
        ),
        live_slippage_pct=max(
            0.0,
            min(10.0, _env_float("OPENJARVIS_CRYPTO_TRADER_LIVE_SLIPPAGE_PCT", 2.0)),
        ),
        live_max_actions_per_tick=max(
            1, int(_env_float("OPENJARVIS_CRYPTO_TRADER_LIVE_MAX_ACTIONS_PER_TICK", 1))
        ),
        live_max_open_positions=max(
            1, int(_env_float("OPENJARVIS_CRYPTO_TRADER_LIVE_MAX_OPEN_POSITIONS", 2))
        ),
        live_entry_momentum_pct=max(
            Decimal("0.00005"),
            _env_decimal("OPENJARVIS_CRYPTO_TRADER_LIVE_ENTRY_MOMENTUM_PCT", "0.002"),
        ),
        live_exit_momentum_pct=max(
            Decimal("0.00005"),
            _env_decimal("OPENJARVIS_CRYPTO_TRADER_LIVE_EXIT_MOMENTUM_PCT", "0.002"),
        ),
        live_learning_enabled=_env_bool(
            "OPENJARVIS_CRYPTO_TRADER_LIVE_LEARNING_ENABLED", True
        ),
        live_learning_hold_streak=max(
            2, int(_env_float("OPENJARVIS_CRYPTO_TRADER_LIVE_LEARNING_HOLD_STREAK", 8))
        ),
        live_learning_step_pct=max(
            Decimal("0.0001"),
            _env_decimal("OPENJARVIS_CRYPTO_TRADER_LIVE_LEARNING_STEP_PCT", "0.0002"),
        ),
        live_learning_min_momentum_pct=max(
            Decimal("0.00005"),
            _env_decimal(
                "OPENJARVIS_CRYPTO_TRADER_LIVE_LEARNING_MIN_MOMENTUM_PCT", "0.0008"
            ),
        ),
        live_learning_max_momentum_pct=max(
            Decimal("0.0015"),
            _env_decimal(
                "OPENJARVIS_CRYPTO_TRADER_LIVE_LEARNING_MAX_MOMENTUM_PCT", "0.006"
            ),
        ),
        live_min_actions_per_hour=max(
            0,
            int(_env_float("OPENJARVIS_CRYPTO_TRADER_LIVE_MIN_ACTIONS_PER_HOUR", 2)),
        ),
        live_activity_boost_step_pct=max(
            Decimal("0.0001"),
            _env_decimal(
                "OPENJARVIS_CRYPTO_TRADER_LIVE_ACTIVITY_BOOST_STEP_PCT", "0.00025"
            ),
        ),
        live_score_momentum_weight=_env_decimal(
            "OPENJARVIS_CRYPTO_TRADER_LIVE_SCORE_MOMENTUM_WEIGHT", "1.0"
        ),
        live_score_news_weight=_env_decimal(
            "OPENJARVIS_CRYPTO_TRADER_LIVE_SCORE_NEWS_WEIGHT", "0.25"
        ),
        live_score_regime_weight=_env_decimal(
            "OPENJARVIS_CRYPTO_TRADER_LIVE_SCORE_REGIME_WEIGHT", "0.5"
        ),
        live_score_position_bonus=_env_decimal(
            "OPENJARVIS_CRYPTO_TRADER_LIVE_SCORE_POSITION_BONUS", "0.35"
        ),
        live_min_opportunity_score=_env_decimal(
            "OPENJARVIS_CRYPTO_TRADER_LIVE_MIN_OPPORTUNITY_SCORE", "0.15"
        ),
        live_opportunity_relax_after_holds=max(
            1,
            int(
                _env_float(
                    "OPENJARVIS_CRYPTO_TRADER_LIVE_OPPORTUNITY_RELAX_AFTER_HOLDS",
                    12,
                )
            ),
        ),
        live_opportunity_relax_step_pct=max(
            Decimal("0"),
            _env_decimal(
                "OPENJARVIS_CRYPTO_TRADER_LIVE_OPPORTUNITY_RELAX_STEP_PCT",
                "0.00005",
            ),
        ),
        live_risk_daily_max_drawdown_pct=max(
            Decimal("0.005"),
            _env_decimal(
                "OPENJARVIS_CRYPTO_TRADER_LIVE_RISK_DAILY_MAX_DRAWDOWN_PCT", "0.035"
            ),
        ),
        live_risk_max_consecutive_losses=max(
            1,
            int(
                _env_float(
                    "OPENJARVIS_CRYPTO_TRADER_LIVE_RISK_MAX_CONSECUTIVE_LOSSES", 3
                )
            ),
        ),
        live_risk_cooldown_minutes=max(
            1,
            int(_env_float("OPENJARVIS_CRYPTO_TRADER_LIVE_RISK_COOLDOWN_MINUTES", 60)),
        ),
        live_micro_tx_fee_ratio=max(
            Decimal("0"),
            _env_decimal("OPENJARVIS_CRYPTO_TRADER_LIVE_MICRO_TX_FEE_RATIO", "0.003"),
        ),
        live_micro_gas_buffer_usdc=max(
            Decimal("0"),
            _env_decimal("OPENJARVIS_CRYPTO_TRADER_LIVE_MICRO_GAS_BUFFER_USDC", "0.35"),
        ),
        live_micro_min_net_edge_pct=max(
            Decimal("0.000001"),
            _env_decimal(
                "OPENJARVIS_CRYPTO_TRADER_LIVE_MICRO_MIN_NET_EDGE_PCT", "0.004"
            ),
        ),
        live_micro_relax_after_holds=max(
            1,
            int(
                _env_float(
                    "OPENJARVIS_CRYPTO_TRADER_LIVE_MICRO_RELAX_AFTER_HOLDS",
                    10,
                )
            ),
        ),
        live_micro_relax_step_pct=max(
            Decimal("0"),
            _env_decimal(
                "OPENJARVIS_CRYPTO_TRADER_LIVE_MICRO_RELAX_STEP_PCT", "0.00001"
            ),
        ),
        live_micro_min_floor_pct=max(
            Decimal("0.000001"),
            _env_decimal(
                "OPENJARVIS_CRYPTO_TRADER_LIVE_MICRO_MIN_FLOOR_PCT", "0.000005"
            ),
        ),
        news_filter_enabled=_env_bool(
            "OPENJARVIS_CRYPTO_TRADER_NEWS_FILTER_ENABLED", True
        ),
        news_feed_url=os.environ.get(
            "OPENJARVIS_CRYPTO_TRADER_NEWS_FEED_URL",
            "http://127.0.0.1:5055/v1/news/feed?kind=all&limit=60",
        ).strip(),
        news_max_items=max(
            1, int(_env_float("OPENJARVIS_CRYPTO_TRADER_NEWS_MAX_ITEMS", 40))
        ),
        news_lookback_hours=max(
            1, int(_env_float("OPENJARVIS_CRYPTO_TRADER_NEWS_LOOKBACK_HOURS", 24))
        ),
        news_block_buy_score=min(
            0, int(_env_float("OPENJARVIS_CRYPTO_TRADER_NEWS_BLOCK_BUY_SCORE", -2))
        ),
        news_force_exit_score=min(
            0, int(_env_float("OPENJARVIS_CRYPTO_TRADER_NEWS_FORCE_EXIT_SCORE", -4))
        ),
        min_gas_eth=max(
            Decimal("0"),
            _env_decimal("OPENJARVIS_CRYPTO_TRADER_MIN_GAS_ETH", "0.002"),
        ),
        gas_warning_eth=max(
            Decimal("0"),
            _env_decimal("OPENJARVIS_CRYPTO_TRADER_GAS_WARNING_ETH", "0.004"),
        ),
        gas_refill_enabled=_env_bool(
            "OPENJARVIS_CRYPTO_TRADER_GAS_REFILL_ENABLED", True
        ),
        gas_refill_target_eth=max(
            Decimal("0"),
            _env_decimal("OPENJARVIS_CRYPTO_TRADER_GAS_REFILL_TARGET_ETH", "0.005"),
        ),
        gas_refill_usdc_budget=max(
            Decimal("0"),
            _env_decimal("OPENJARVIS_CRYPTO_TRADER_GAS_REFILL_USDC_BUDGET", "5"),
        ),
        ai_ollama_host=os.environ.get(
            "OPENJARVIS_CRYPTO_TRADER_AI_OLLAMA_HOST", "http://192.168.1.9:11434"
        ).strip(),
        ai_ollama_model=os.environ.get(
            "OPENJARVIS_CRYPTO_TRADER_AI_OLLAMA_MODEL", "agent-default:latest"
        ).strip(),
    )


def _cfg_with_overrides(base: TraderConfig, overrides: Dict[str, Any]) -> TraderConfig:
    """Apply safe per-agent overrides to TraderConfig.

    Overrides are optional and intended for managed-agent config_json values.
    """

    def _pick(*names: str) -> Any:
        for n in names:
            if n in overrides:
                return overrides[n]
        return None

    def _bool(v: Any, default: bool) -> bool:
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in {"1", "true", "yes", "on"}

    def _int(v: Any, default: int, floor: int = 0) -> int:
        if v is None:
            return default
        try:
            return max(floor, int(v))
        except (TypeError, ValueError):
            return default

    def _int_any(v: Any, default: int) -> int:
        if v is None:
            return default
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _dec(
        v: Any,
        default: Decimal,
        *,
        lo: Decimal | None = None,
        hi: Decimal | None = None,
    ) -> Decimal:
        if v is None:
            return default
        out = _as_decimal(v, str(default))
        if lo is not None and out < lo:
            out = lo
        if hi is not None and out > hi:
            out = hi
        return out

    strategy_mode = (
        str(_pick("strategy_mode", "crypto_strategy_mode") or base.strategy_mode)
        .strip()
        .lower()
    )
    if strategy_mode not in {"wallet_only", "paper", "live"}:
        strategy_mode = base.strategy_mode

    live_execution_backend = (
        str(
            _pick("live_execution_backend", "crypto_live_execution_backend")
            or base.live_execution_backend
        )
        .strip()
        .lower()
    )
    if live_execution_backend not in {
        "wallet",
        "binance_spot",
        "metamask_perps",
        "hyperliquid_perps",
    }:
        live_execution_backend = base.live_execution_backend

    live_wallet_fallback_backend = (
        str(
            _pick(
                "live_wallet_fallback_backend",
                "crypto_live_wallet_fallback_backend",
            )
            or base.live_wallet_fallback_backend
        )
        .strip()
        .lower()
    )
    if live_wallet_fallback_backend not in {
        "none",
        "wallet",
        "binance_spot",
        "metamask_perps",
        "hyperliquid_perps",
    }:
        live_wallet_fallback_backend = base.live_wallet_fallback_backend

    return TraderConfig(
        binance_address=str(
            _pick("binance_address", "crypto_binance_address") or base.binance_address
        ).strip(),
        autosweep_enabled=_bool(
            _pick("autosweep_enabled", "crypto_autosweep_enabled"),
            base.autosweep_enabled,
        ),
        base_capital_usdc=_dec(
            _pick("base_capital_usdc", "crypto_base_capital_usdc"),
            base.base_capital_usdc,
            lo=Decimal("0"),
        ),
        reserve_usdc=_dec(
            _pick("reserve_usdc", "crypto_reserve_usdc"),
            base.reserve_usdc,
            lo=Decimal("0"),
        ),
        min_sweep_usdc=_dec(
            _pick("min_sweep_usdc", "crypto_min_sweep_usdc"),
            base.min_sweep_usdc,
            lo=Decimal("0"),
        ),
        sweep_cooldown_sec=_int(
            _pick("sweep_cooldown_sec", "crypto_sweep_cooldown_sec"),
            base.sweep_cooldown_sec,
            floor=0,
        ),
        max_sweeps_per_day=_int(
            _pick("max_sweeps_per_day", "crypto_max_sweeps_per_day"),
            base.max_sweeps_per_day,
            floor=1,
        ),
        confirm_code=str(
            _pick("confirm_code", "crypto_confirm_code") or base.confirm_code
        ).strip(),
        use_ai=_bool(_pick("use_ai", "crypto_use_ai"), base.use_ai),
        ai_min_confidence=max(
            0.0,
            min(
                1.0,
                float(
                    _pick("ai_min_confidence", "crypto_ai_min_confidence")
                    or base.ai_min_confidence
                ),
            ),
        ),
        default_payout_ratio=_dec(
            _pick("default_payout_ratio", "crypto_default_payout_ratio"),
            base.default_payout_ratio,
            lo=Decimal("0"),
            hi=Decimal("1"),
        ),
        ai_max_payout_ratio=_dec(
            _pick("ai_max_payout_ratio", "crypto_ai_max_payout_ratio"),
            base.ai_max_payout_ratio,
            lo=Decimal("0"),
            hi=Decimal("1"),
        ),
        strategy_mode=strategy_mode,
        paper_symbol=str(
            _pick("paper_symbol", "crypto_paper_symbol") or base.paper_symbol
        )
        .strip()
        .upper(),
        paper_notional_usdc=_dec(
            _pick("paper_notional_usdc", "crypto_paper_notional_usdc"),
            base.paper_notional_usdc,
            lo=Decimal("0"),
        ),
        paper_take_profit_pct=_dec(
            _pick("paper_take_profit_pct", "crypto_paper_take_profit_pct"),
            base.paper_take_profit_pct,
            lo=Decimal("0.001"),
        ),
        paper_stop_loss_pct=_dec(
            _pick("paper_stop_loss_pct", "crypto_paper_stop_loss_pct"),
            base.paper_stop_loss_pct,
            lo=Decimal("0.001"),
        ),
        paper_allow_real_sweep=_bool(
            _pick("paper_allow_real_sweep", "crypto_paper_allow_real_sweep"),
            base.paper_allow_real_sweep,
        ),
        live_symbol=str(_pick("live_symbol", "crypto_live_symbol") or base.live_symbol)
        .strip()
        .upper(),
        live_symbols=str(
            _pick("live_symbols", "crypto_live_symbols") or base.live_symbols
        )
        .strip()
        .upper(),
        live_execution_backend=live_execution_backend,
        live_wallet_fallback_backend=live_wallet_fallback_backend,
        live_perps_leverage=_int(
            _pick("live_perps_leverage", "crypto_live_perps_leverage"),
            base.live_perps_leverage,
            floor=1,
        ),
        live_binance_test_order=_bool(
            _pick("live_binance_test_order", "crypto_live_binance_test_order"),
            base.live_binance_test_order,
        ),
        live_wallet_network=str(
            _pick("live_wallet_network", "crypto_live_wallet_network")
            or base.live_wallet_network
        )
        .strip()
        .lower(),
        live_wallet_networks=str(
            _pick("live_wallet_networks", "crypto_live_wallet_networks")
            or base.live_wallet_networks
        )
        .strip()
        .lower(),
        live_wallet_watch_symbols=str(
            _pick("live_wallet_watch_symbols", "crypto_live_wallet_watch_symbols")
            or base.live_wallet_watch_symbols
        )
        .strip()
        .upper(),
        live_notional_usdc=_dec(
            _pick("live_notional_usdc", "crypto_live_notional_usdc"),
            base.live_notional_usdc,
            lo=Decimal("0"),
        ),
        growth_enabled=_bool(
            _pick("growth_enabled", "crypto_growth_enabled"),
            base.growth_enabled,
        ),
        growth_reinvest_ratio=_dec(
            _pick("growth_reinvest_ratio", "crypto_growth_reinvest_ratio"),
            base.growth_reinvest_ratio,
            lo=Decimal("0"),
            hi=Decimal("1"),
        ),
        growth_max_notional_usdc=_dec(
            _pick("growth_max_notional_usdc", "crypto_growth_max_notional_usdc"),
            base.growth_max_notional_usdc,
            lo=Decimal("0"),
        ),
        live_take_profit_pct=_dec(
            _pick("live_take_profit_pct", "crypto_live_take_profit_pct"),
            base.live_take_profit_pct,
            lo=Decimal("0.001"),
        ),
        live_stop_loss_pct=_dec(
            _pick("live_stop_loss_pct", "crypto_live_stop_loss_pct"),
            base.live_stop_loss_pct,
            lo=Decimal("0.001"),
        ),
        live_max_position_minutes=_int(
            _pick("live_max_position_minutes", "crypto_live_max_position_minutes"),
            base.live_max_position_minutes,
            floor=1,
        ),
        live_timeout_exit_pnl_pct=_dec(
            _pick("live_timeout_exit_pnl_pct", "crypto_live_timeout_exit_pnl_pct"),
            base.live_timeout_exit_pnl_pct,
        ),
        live_min_trade_usdc=_dec(
            _pick("live_min_trade_usdc", "crypto_live_min_trade_usdc"),
            base.live_min_trade_usdc,
            lo=Decimal("0"),
        ),
        live_slippage_pct=max(
            0.0,
            min(
                10.0,
                float(
                    _pick("live_slippage_pct", "crypto_live_slippage_pct")
                    or base.live_slippage_pct
                ),
            ),
        ),
        live_max_actions_per_tick=max(
            1,
            int(
                _pick("live_max_actions_per_tick", "crypto_live_max_actions_per_tick")
                or base.live_max_actions_per_tick
            ),
        ),
        live_max_open_positions=max(
            1,
            int(
                _pick("live_max_open_positions", "crypto_live_max_open_positions")
                or base.live_max_open_positions
            ),
        ),
        live_entry_momentum_pct=_dec(
            _pick("live_entry_momentum_pct", "crypto_live_entry_momentum_pct"),
            base.live_entry_momentum_pct,
            lo=Decimal("0.00005"),
        ),
        live_exit_momentum_pct=_dec(
            _pick("live_exit_momentum_pct", "crypto_live_exit_momentum_pct"),
            base.live_exit_momentum_pct,
            lo=Decimal("0.00005"),
        ),
        live_learning_enabled=_bool(
            _pick("live_learning_enabled", "crypto_live_learning_enabled"),
            base.live_learning_enabled,
        ),
        live_learning_hold_streak=_int(
            _pick("live_learning_hold_streak", "crypto_live_learning_hold_streak"),
            base.live_learning_hold_streak,
            floor=2,
        ),
        live_learning_step_pct=_dec(
            _pick("live_learning_step_pct", "crypto_live_learning_step_pct"),
            base.live_learning_step_pct,
            lo=Decimal("0.0001"),
        ),
        live_learning_min_momentum_pct=_dec(
            _pick(
                "live_learning_min_momentum_pct",
                "crypto_live_learning_min_momentum_pct",
            ),
            base.live_learning_min_momentum_pct,
            lo=Decimal("0.00005"),
        ),
        live_learning_max_momentum_pct=_dec(
            _pick(
                "live_learning_max_momentum_pct",
                "crypto_live_learning_max_momentum_pct",
            ),
            base.live_learning_max_momentum_pct,
            lo=Decimal("0.0015"),
        ),
        live_min_actions_per_hour=_int(
            _pick("live_min_actions_per_hour", "crypto_live_min_actions_per_hour"),
            base.live_min_actions_per_hour,
            floor=0,
        ),
        live_activity_boost_step_pct=_dec(
            _pick(
                "live_activity_boost_step_pct", "crypto_live_activity_boost_step_pct"
            ),
            base.live_activity_boost_step_pct,
            lo=Decimal("0.0001"),
        ),
        live_score_momentum_weight=_dec(
            _pick("live_score_momentum_weight", "crypto_live_score_momentum_weight"),
            base.live_score_momentum_weight,
            lo=Decimal("0"),
        ),
        live_score_news_weight=_dec(
            _pick("live_score_news_weight", "crypto_live_score_news_weight"),
            base.live_score_news_weight,
            lo=Decimal("0"),
        ),
        live_score_regime_weight=_dec(
            _pick("live_score_regime_weight", "crypto_live_score_regime_weight"),
            base.live_score_regime_weight,
            lo=Decimal("0"),
        ),
        live_score_position_bonus=_dec(
            _pick("live_score_position_bonus", "crypto_live_score_position_bonus"),
            base.live_score_position_bonus,
            lo=Decimal("0"),
        ),
        live_min_opportunity_score=_dec(
            _pick("live_min_opportunity_score", "crypto_live_min_opportunity_score"),
            base.live_min_opportunity_score,
            lo=Decimal("-1"),
        ),
        live_opportunity_relax_after_holds=_int(
            _pick(
                "live_opportunity_relax_after_holds",
                "crypto_live_opportunity_relax_after_holds",
            ),
            base.live_opportunity_relax_after_holds,
            floor=1,
        ),
        live_opportunity_relax_step_pct=_dec(
            _pick(
                "live_opportunity_relax_step_pct",
                "crypto_live_opportunity_relax_step_pct",
            ),
            base.live_opportunity_relax_step_pct,
            lo=Decimal("0"),
        ),
        live_risk_daily_max_drawdown_pct=_dec(
            _pick(
                "live_risk_daily_max_drawdown_pct",
                "crypto_live_risk_daily_max_drawdown_pct",
            ),
            base.live_risk_daily_max_drawdown_pct,
            lo=Decimal("0.005"),
        ),
        live_risk_max_consecutive_losses=_int(
            _pick(
                "live_risk_max_consecutive_losses",
                "crypto_live_risk_max_consecutive_losses",
            ),
            base.live_risk_max_consecutive_losses,
            floor=1,
        ),
        live_risk_cooldown_minutes=_int(
            _pick("live_risk_cooldown_minutes", "crypto_live_risk_cooldown_minutes"),
            base.live_risk_cooldown_minutes,
            floor=1,
        ),
        live_micro_tx_fee_ratio=_dec(
            _pick("live_micro_tx_fee_ratio", "crypto_live_micro_tx_fee_ratio"),
            base.live_micro_tx_fee_ratio,
            lo=Decimal("0"),
        ),
        live_micro_gas_buffer_usdc=_dec(
            _pick("live_micro_gas_buffer_usdc", "crypto_live_micro_gas_buffer_usdc"),
            base.live_micro_gas_buffer_usdc,
            lo=Decimal("0"),
        ),
        live_micro_min_net_edge_pct=_dec(
            _pick("live_micro_min_net_edge_pct", "crypto_live_micro_min_net_edge_pct"),
            base.live_micro_min_net_edge_pct,
            lo=Decimal("0.000001"),
        ),
        live_micro_relax_after_holds=_int(
            _pick(
                "live_micro_relax_after_holds",
                "crypto_live_micro_relax_after_holds",
            ),
            base.live_micro_relax_after_holds,
            floor=1,
        ),
        live_micro_relax_step_pct=_dec(
            _pick(
                "live_micro_relax_step_pct",
                "crypto_live_micro_relax_step_pct",
            ),
            base.live_micro_relax_step_pct,
            lo=Decimal("0"),
        ),
        live_micro_min_floor_pct=_dec(
            _pick(
                "live_micro_min_floor_pct",
                "crypto_live_micro_min_floor_pct",
            ),
            base.live_micro_min_floor_pct,
            lo=Decimal("0.000001"),
        ),
        news_filter_enabled=_bool(
            _pick("news_filter_enabled", "crypto_news_filter_enabled"),
            base.news_filter_enabled,
        ),
        news_feed_url=str(
            _pick("news_feed_url", "crypto_news_feed_url") or base.news_feed_url
        ).strip(),
        news_max_items=_int(
            _pick("news_max_items", "crypto_news_max_items"),
            base.news_max_items,
            floor=1,
        ),
        news_lookback_hours=_int(
            _pick("news_lookback_hours", "crypto_news_lookback_hours"),
            base.news_lookback_hours,
            floor=1,
        ),
        news_block_buy_score=min(
            0,
            _int_any(
                _pick("news_block_buy_score", "crypto_news_block_buy_score"),
                base.news_block_buy_score,
            ),
        ),
        news_force_exit_score=min(
            0,
            _int_any(
                _pick("news_force_exit_score", "crypto_news_force_exit_score"),
                base.news_force_exit_score,
            ),
        ),
        min_gas_eth=_dec(
            _pick("min_gas_eth", "crypto_min_gas_eth"),
            base.min_gas_eth,
            lo=Decimal("0"),
        ),
        gas_warning_eth=_dec(
            _pick("gas_warning_eth", "crypto_gas_warning_eth"),
            base.gas_warning_eth,
            lo=Decimal("0"),
        ),
        gas_refill_enabled=_bool(
            _pick("gas_refill_enabled", "crypto_gas_refill_enabled"),
            base.gas_refill_enabled,
        ),
        gas_refill_target_eth=_dec(
            _pick("gas_refill_target_eth", "crypto_gas_refill_target_eth"),
            base.gas_refill_target_eth,
            lo=Decimal("0"),
        ),
        gas_refill_usdc_budget=_dec(
            _pick("gas_refill_usdc_budget", "crypto_gas_refill_usdc_budget"),
            base.gas_refill_usdc_budget,
            lo=Decimal("0"),
        ),
        ai_ollama_host=str(
            _pick("ai_ollama_host", "crypto_ai_ollama_host") or base.ai_ollama_host
        ).strip(),
        ai_ollama_model=str(
            _pick("ai_ollama_model", "crypto_ai_ollama_model", "model")
            or base.ai_ollama_model
        ).strip(),
    )


def _load_state() -> Dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return _default_state()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            merged = _default_state()
            merged.update(data)
            return merged
    except Exception:
        pass
    return _default_state()


def _save_state(state: Dict[str, Any]) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, ensure_ascii=True, indent=2)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=p.parent,
            prefix=f".{p.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, p)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _as_decimal(value: Any, default: str = "0") -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal(default)


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract a JSON object from model output (raw or fenced/verbose)."""
    if not text:
        return {}
    raw = text.strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        parsed = json.loads(raw[start : end + 1])
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _fetch_binance_price(symbol: str) -> Decimal:
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        with urlopen(url, timeout=8) as resp:  # nosec B310: public HTTPS API fetch
            payload = json.loads(resp.read().decode("utf-8"))
    except (URLError, TimeoutError, ValueError) as exc:
        raise RuntimeError(f"price_fetch_failed:{type(exc).__name__}") from exc

    price = _as_decimal(payload.get("price", "0"))
    if price <= Decimal("0"):
        raise RuntimeError("price_invalid")
    return price


def _asset_from_symbol(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    for q in ("USDT", "USDC", "USD"):
        if s.endswith(q) and len(s) > len(q):
            return s[: -len(q)]
    return s


_WALLET_TOKEN_SYMBOL_ALIASES = {
    "POL": "MATIC",
}


def _wallet_token_symbol(asset: str) -> str:
    raw = (asset or "").strip().upper()
    if not raw:
        return ""
    return _WALLET_TOKEN_SYMBOL_ALIASES.get(raw, raw)


def _quote_from_symbol(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    for q in ("USDT", "USDC", "USD"):
        if s.endswith(q) and len(s) > len(q):
            return q
    return "USDT"


def _live_symbols(cfg: TraderConfig) -> list[str]:
    symbols: list[str] = []
    raw = (cfg.live_symbols or "").strip()
    if raw:
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if not symbols and cfg.live_symbol:
        symbols = [cfg.live_symbol.strip().upper()]
    out: list[str] = []
    seen: set[str] = set()
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


_WALLET_NETWORK_ALIASES = {
    "base-mainnet": "base",
    "bsc": "bnb",
    "binance": "bnb",
    "binance-smart-chain": "bnb",
    "matic": "polygon",
    "polygon-pos": "polygon",
    "eth": "ethereum",
    "mainnet": "ethereum",
    "optimism": "op",
    "arb": "arbitrum",
    "avax": "avalanche",
}


def _normalize_wallet_network(value: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return ""
    return _WALLET_NETWORK_ALIASES.get(raw, raw)


def _live_wallet_networks(cfg: TraderConfig) -> list[str]:
    raw_items: list[str] = []
    if cfg.live_wallet_networks:
        raw_items.extend(
            [
                part.strip()
                for part in cfg.live_wallet_networks.split(",")
                if part.strip()
            ]
        )
    if cfg.live_wallet_network:
        raw_items.append(cfg.live_wallet_network)
    if not raw_items:
        raw_items = ["base"]

    out: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        normalized = _normalize_wallet_network(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _wallet_watch_symbols(cfg: TraderConfig) -> list[str]:
    raw = (cfg.live_wallet_watch_symbols or "").strip()
    symbols: list[str] = []
    if raw:
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if not symbols:
        symbols = [_asset_from_symbol(s) for s in _live_symbols(cfg)]
        symbols.extend(["USDC", "ETH", "BTC", "BNB", "MATIC", "WETH", "CBTC"])

    out: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


_NEGATIVE_NEWS_WORDS = {
    "hack",
    "hacked",
    "exploit",
    "breach",
    "ban",
    "lawsuit",
    "sec",
    "crime",
    "liquidation",
    "outage",
    "downtime",
    "recession",
    "inflation",
    "rate hike",
    "war",
    "tariff",
    "sanction",
    "dump",
    "crash",
    "bankrupt",
    "bankruptcy",
}
_POSITIVE_NEWS_WORDS = {
    "approval",
    "approved",
    "etf",
    "adoption",
    "partnership",
    "upgrade",
    "launch",
    "inflow",
    "bullish",
    "rally",
    "breakout",
    "rate cut",
    "eases",
    "integration",
    "record high",
}


def _news_term_count(text: str, terms: set[str]) -> int:
    """Count complete sentiment terms, never arbitrary substrings."""
    normalized = str(text or "").lower()
    return sum(
        1
        for term in terms
        if re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", normalized)
    )


def _asset_aliases(asset: str) -> set[str]:
    a = (asset or "").upper()
    aliases: dict[str, set[str]] = {
        "ETH": {"eth", "ethereum", "ether", "base"},
        "BTC": {"btc", "bitcoin", "cbbtc", "spot bitcoin"},
        "SOL": {"sol", "solana"},
        "POL": {"pol", "matic", "polygon"},
        "MATIC": {"matic", "pol", "polygon"},
    }
    return aliases.get(a, {a.lower()})


def _sma_decimal(series: list[str], window: int) -> Decimal:
    if window <= 0 or len(series) < window:
        return Decimal("0")
    vals = [_as_decimal(v, "0") for v in series[-window:]]
    total = sum(vals, Decimal("0"))
    return total / Decimal(str(window))


def _pattern_forecast(series: list[str]) -> Decimal:
    """Bounded multi-horizon directional forecast in [-1, 1].

    Returns are normalized by recent realized volatility so the same signal
    scale works across BTC, majors, and smaller tokens.  Multiple horizons
    must broadly agree before the score becomes large.
    """
    values = [_as_decimal(value, "0") for value in series[-60:]]
    values = [value for value in values if value > Decimal("0")]
    if len(values) < 22:
        return Decimal("0")

    returns: list[Decimal] = []
    for previous, current in zip(values, values[1:]):
        if previous > Decimal("0"):
            returns.append((current - previous) / previous)
    if len(returns) < 20:
        return Decimal("0")

    mean = sum(returns[-20:], Decimal("0")) / Decimal("20")
    variance = sum(
        ((item - mean) * (item - mean) for item in returns[-20:]),
        Decimal("0"),
    ) / Decimal("20")
    volatility = Decimal(str(float(variance) ** 0.5))
    scale = max(volatility, Decimal("0.00015"))

    current = values[-1]
    horizon_scores: list[tuple[Decimal, Decimal]] = []
    for horizon, weight in ((1, "0.15"), (3, "0.25"), (8, "0.30"), (21, "0.30")):
        past = values[-1 - horizon]
        horizon_return = (current - past) / past
        normalized = horizon_return / (scale * Decimal(str(horizon)) ** Decimal("0.5"))
        normalized = max(Decimal("-2"), min(Decimal("2"), normalized))
        horizon_scores.append((normalized, Decimal(weight)))

    combined = sum((score * weight for score, weight in horizon_scores), Decimal("0"))
    # Divide by two because each component is clipped to +/-2.
    return max(Decimal("-1"), min(Decimal("1"), combined / Decimal("2")))


def _pattern_reliability(series: list[str]) -> Decimal:
    """Walk-forward confidence for the pattern model, bounded to [0, 1]."""
    values = list(series[-120:])
    if len(values) < 45:
        return Decimal("0")
    signals = 0
    hits = 0
    threshold = Decimal("0.22")
    for index in range(22, len(values) - 1):
        forecast = _pattern_forecast(values[: index + 1])
        if abs(forecast) < threshold:
            continue
        current = _as_decimal(values[index], "0")
        following = _as_decimal(values[index + 1], "0")
        if current <= Decimal("0") or following == current:
            continue
        signals += 1
        hits += int((forecast > Decimal("0")) == (following > current))
    if signals < 20:
        return Decimal("0")
    accuracy = Decimal(str(hits)) / Decimal(str(signals))
    return max(
        Decimal("0"),
        min(Decimal("1"), (accuracy - Decimal("0.52")) / Decimal("0.13")),
    )


def _parse_iso_datetime(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


@AgentRegistry.register("crypto_trader")
class CryptoTraderAgent(BaseAgent):
    """Managed-agent strategy that sweeps realized USDC profits to Binance."""

    agent_id = "crypto_trader"
    requires_engine = False
    requires_model = False

    def _ollama_generate(self, cfg: "TraderConfig", messages: list) -> dict:
        """Call Ollama chat API directly at cfg.ai_ollama_host.

        Returns dict with 'content' key (matching self._generate() shape).
        Falls back to self._generate() if Ollama call fails.
        """
        if not cfg.ai_ollama_host or not cfg.ai_ollama_model:
            return {}
        try:
            payload = {
                "model": cfg.ai_ollama_model,
                "messages": [
                    {
                        "role": getattr(m, "role", "user").value
                        if hasattr(getattr(m, "role", None), "value")
                        else str(getattr(m, "role", "user")),
                        "content": str(getattr(m, "content", m)),
                    }
                    for m in messages
                ],
                "stream": False,
                "options": {"temperature": 0},
            }
            with __import__("httpx").Client(timeout=30) as client:
                resp = client.post(
                    f"{cfg.ai_ollama_host.rstrip('/')}/api/chat",
                    json=payload,
                )
            resp.raise_for_status()
            data = resp.json()
            content = (data.get("message") or {}).get("content", "")
            return {"content": content}
        except Exception as exc:
            return {"content": "", "_ollama_error": str(exc)}

    def _ai_trade_signal(
        self,
        cfg: "TraderConfig",
        *,
        symbol: str,
        price: Decimal,
        prev_price: Decimal,
        has_position: bool,
        news_context: str = "",
    ) -> Dict[str, Any]:
        if not cfg.use_ai:
            return {
                "used": False,
                "action": "hold",
                "confidence": 0.0,
                "reason": "ai_disabled",
            }
        try:
            from openjarvis.core.types import Message, Role

            messages = [
                Message(
                    role=Role.SYSTEM,
                    content=(
                        "You are a strict trading signal filter for one-symbol spot paper trading. "
                        "Return JSON only with keys: action (buy|sell|hold), confidence (0..1), reason."
                    ),
                ),
                Message(
                    role=Role.USER,
                    content=(
                        f"symbol={symbol}\n"
                        f"price_now={price}\n"
                        f"price_prev={prev_price}\n"
                        f"has_position={has_position}\n"
                        f"news_context={news_context or 'none'}\n"
                        "Goal: favor risk control over frequency."
                    ),
                ),
            ]
            result = self._ollama_generate(cfg, messages)
            if result.get("_ollama_error") and not (
                getattr(self, "_engine", None) and getattr(self, "_model", "")
            ):
                return {
                    "used": False,
                    "action": "hold",
                    "confidence": 0.0,
                    "reason": "engine_or_model_unavailable",
                }
            if not result.get("content") and (
                getattr(self, "_engine", None) and getattr(self, "_model", "")
            ):
                result = self._generate(messages)
            payload = _extract_json_object(str(result.get("content", "")))
            action = str(payload.get("action", "hold")).strip().lower()
            if action not in {"buy", "sell", "hold"}:
                action = "hold"
            conf = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
            reason = str(payload.get("reason", "ai_ok"))[:280]
            return {
                "used": True,
                "action": action,
                "confidence": conf,
                "reason": reason,
            }
        except Exception as exc:
            return {
                "used": False,
                "action": "hold",
                "confidence": 0.0,
                "reason": f"ai_error:{type(exc).__name__}",
            }

    def _fetch_news_feed(self, cfg: "TraderConfig") -> Dict[str, Any]:
        if not cfg.news_filter_enabled or not cfg.news_feed_url:
            return {"items": [], "error": "news_disabled"}
        try:
            headers: Dict[str, str] = {}
            internal_api_key = os.environ.get("OPENJARVIS_API_KEY", "").strip()
            if not internal_api_key:
                try:
                    import tomllib

                    config_path = Path.home() / ".openjarvis" / "config.toml"
                    with config_path.open("rb") as config_file:
                        raw_config = tomllib.load(config_file)
                    internal_api_key = str(
                        raw_config.get("server", {}).get("auth", {}).get("api_key", "")
                    ).strip()
                except (OSError, ValueError, TypeError):
                    internal_api_key = ""
            if internal_api_key:
                headers["Authorization"] = f"Bearer {internal_api_key}"
            with __import__("httpx").Client(timeout=8) as client:
                resp = client.get(cfg.news_feed_url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
            items = payload.get("items") if isinstance(payload, dict) else []
            if not isinstance(items, list):
                items = []
            return {"items": items[: cfg.news_max_items]}
        except Exception as exc:
            return {"items": [], "error": f"news_fetch_error:{type(exc).__name__}"}

    def _news_guard_for_symbol(
        self,
        cfg: "TraderConfig",
        *,
        symbol: str,
        news_items: list[dict[str, Any]],
    ) -> Dict[str, Any]:
        if not cfg.news_filter_enabled:
            return {
                "enabled": False,
                "relevant_items": 0,
                "score": 0,
                "summary": "disabled",
                "block_buy": False,
                "force_exit": False,
            }

        asset = _asset_from_symbol(symbol)
        aliases = _asset_aliases(asset)
        lookback_sec = max(3600, cfg.news_lookback_hours * 3600)
        now = datetime.now(timezone.utc)
        score = 0
        relevant = 0
        snippets: list[str] = []

        for item in news_items[: cfg.news_max_items]:
            if not isinstance(item, dict):
                continue
            published = _parse_iso_datetime(str(item.get("published_at", "")))
            if published and (now - published).total_seconds() > lookback_sec:
                continue

            title = str(item.get("title", ""))
            summary = str(item.get("summary", ""))
            text = f"{title} {summary}".lower()

            asset_match = any(
                re.search(rf"\\b{re.escape(a)}\\b", text) for a in aliases
            )
            macro_match = any(
                k in text
                for k in ("fed", "fomc", "cpi", "inflation", "rates", "crypto")
            )
            if not (asset_match or macro_match):
                continue

            neg = _news_term_count(text, _NEGATIVE_NEWS_WORDS)
            pos = _news_term_count(text, _POSITIVE_NEWS_WORDS)
            if neg == 0 and pos == 0:
                continue
            relevant += 1
            score += pos - neg
            snippets.append(title[:90])

        summary = "none"
        if snippets:
            summary = " | ".join(snippets[:3])

        return {
            "enabled": True,
            "relevant_items": relevant,
            "score": score,
            "summary": summary,
            "block_buy": score <= cfg.news_block_buy_score,
            "force_exit": score <= cfg.news_force_exit_score,
        }

    def _estimate_live_roundtrip_cost_usdc(
        self,
        cfg: "TraderConfig",
        *,
        notional_usdc: Decimal,
        native_eth: Decimal,
    ) -> Decimal:
        if notional_usdc <= Decimal("0"):
            return Decimal("0")
        fee_cost = notional_usdc * max(Decimal("0"), cfg.live_micro_tx_fee_ratio)
        slippage_side = Decimal(str(cfg.live_slippage_pct)) / Decimal("100")
        slippage_cost = notional_usdc * slippage_side * Decimal("2")
        gas_buffer = max(Decimal("0"), cfg.live_micro_gas_buffer_usdc)
        # If gas is near warning threshold, assume a larger effective cost footprint.
        if native_eth < cfg.gas_warning_eth and cfg.gas_refill_usdc_budget > Decimal(
            "0"
        ):
            gas_buffer = max(gas_buffer, cfg.gas_refill_usdc_budget)
        total = fee_cost + slippage_cost + gas_buffer
        return total.quantize(Decimal("0.000001"), rounding=ROUND_DOWN)

    def _wallet_coverage_snapshot(
        self, cfg: TraderConfig, wallet_tool
    ) -> Dict[str, Any]:
        requested_networks = _live_wallet_networks(cfg)
        supported_networks: list[str] = []
        unsupported_networks: list[str] = []
        watch_symbols = _wallet_watch_symbols(cfg)
        network_status: Dict[str, Dict[str, Any]] = {}
        tradable_detected: set[str] = set()
        unsupported_detected: set[str] = set()
        supported_symbol_any_network: set[str] = set()

        for network in requested_networks:
            payload: Dict[str, Any] = {
                "network": network,
                "token_symbols": watch_symbols,
            }
            snap = wallet_tool("portfolio", payload)
            if not snap.get("ok"):
                err_text = str(snap.get("error", "unknown"))
                if "unsupported network" in err_text.lower():
                    unsupported_networks.append(network)
                else:
                    supported_networks.append(network)
                network_status[network] = {
                    "ok": False,
                    "error": err_text,
                }
                continue

            supported_networks.append(network)

            assets = list(snap.get("assets") or [])
            tradable_with_balance: list[str] = []
            unsupported_requested: list[str] = []
            for asset in assets:
                symbol = str((asset or {}).get("symbol", "")).upper()
                kind = str((asset or {}).get("kind", "")).lower()
                has_balance = bool((asset or {}).get("has_balance", False))
                if kind == "unsupported" and symbol:
                    unsupported_requested.append(symbol)
                    unsupported_detected.add(symbol)
                    if symbol == "MATIC":
                        unsupported_detected.add("POL")
                    elif symbol == "POL":
                        unsupported_detected.add("MATIC")
                if (
                    symbol
                    and symbol not in {"NATIVE", "USDC"}
                    and kind != "unsupported"
                    and has_balance
                ):
                    tradable_with_balance.append(symbol)
                    tradable_detected.add(symbol)

            network_status[network] = {
                "ok": True,
                "chain": str(snap.get("chain", network)),
                "positive_asset_count": int(snap.get("positive_asset_count") or 0),
                "tradable_with_balance": sorted(set(tradable_with_balance)),
                "unsupported_requested": sorted(set(unsupported_requested)),
                "supported_token_symbols": list(
                    snap.get("supported_token_symbols") or []
                ),
            }
            for sym in list(snap.get("supported_token_symbols") or []):
                up = str(sym or "").upper()
                if up:
                    supported_symbol_any_network.add(up)

        # A token should be considered unsupported only when we have at least
        # one concrete support map from portfolio snapshots.
        if supported_symbol_any_network:
            for sym in watch_symbols:
                up = str(sym or "").upper()
                if not up or up in {"NATIVE", "USDC"}:
                    continue
                if up not in supported_symbol_any_network:
                    unsupported_detected.add(up)
                    if up == "MATIC":
                        unsupported_detected.add("POL")
                    elif up == "POL":
                        unsupported_detected.add("MATIC")

        return {
            "requested_networks": requested_networks,
            "supported_networks": supported_networks,
            "unsupported_networks": unsupported_networks,
            "watch_symbols": watch_symbols,
            "tradable_tokens_detected": sorted(tradable_detected),
            "unsupported_tokens_detected": sorted(unsupported_detected),
            "network_status": network_status,
            "snapshot_complete": bool(supported_networks),
        }

    def _rank_live_symbols(
        self,
        cfg: "TraderConfig",
        *,
        state: Dict[str, Any],
        symbols: list[str],
        news_items: list[dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        live_last_prices = dict(state.get("live_last_prices") or {})
        live_positions = dict(state.get("live_positions") or {})
        live_price_history_map = dict(state.get("live_price_history_by_symbol") or {})
        ranked: list[Dict[str, Any]] = []

        for symbol in symbols:
            try:
                price = _fetch_binance_price(symbol)
            except Exception:
                continue
            prev_price = _as_decimal(live_last_prices.get(symbol, "0"))
            momentum_pct = Decimal("0")
            if prev_price > Decimal("0"):
                momentum_pct = (price - prev_price) / prev_price

            price_history = list(live_price_history_map.get(symbol) or [])
            price_history.append(str(price))
            if len(price_history) > 120:
                price_history = price_history[-120:]
            sma_fast = _sma_decimal(price_history, 5)
            sma_slow = _sma_decimal(price_history, 20)
            pattern_score = _pattern_forecast(price_history)
            pattern_reliability = _pattern_reliability(price_history)
            effective_pattern_score = pattern_score * pattern_reliability
            regime = "flat"
            regime_score = Decimal("0")
            if sma_fast > Decimal("0") and sma_slow > Decimal("0"):
                if sma_fast >= sma_slow * Decimal("1.0005"):
                    regime = "bull"
                    regime_score = Decimal("1")
                elif sma_fast <= sma_slow * Decimal("0.9995"):
                    regime = "bear"
                    regime_score = Decimal("-1")

            news_guard = self._news_guard_for_symbol(
                cfg, symbol=symbol, news_items=news_items
            )
            news_score_raw = int(news_guard.get("score") or 0)
            news_score = Decimal(str(news_score_raw)) / Decimal("4")
            has_position = bool(live_positions.get(symbol))
            position_bonus = (
                cfg.live_score_position_bonus if has_position else Decimal("0")
            )

            score = (
                momentum_pct * cfg.live_score_momentum_weight
                + news_score * cfg.live_score_news_weight
                + regime_score * cfg.live_score_regime_weight
                + effective_pattern_score
                * _env_decimal(
                    "OPENJARVIS_CRYPTO_TRADER_LIVE_SCORE_PATTERN_WEIGHT", "0.35"
                )
                + position_bonus
            )

            ranked.append(
                {
                    "symbol": symbol,
                    "price": str(price),
                    "prev_price": str(prev_price),
                    "momentum_pct": str(momentum_pct),
                    "regime": regime,
                    "pattern_score": str(pattern_score),
                    "pattern_reliability": str(pattern_reliability),
                    "effective_pattern_score": str(effective_pattern_score),
                    "news_guard": news_guard,
                    "has_position": has_position,
                    "score": str(score),
                }
            )

        ranked.sort(key=lambda x: Decimal(str(x.get("score", "0"))), reverse=True)
        return ranked

    def _paper_trade_step(
        self,
        cfg: TraderConfig,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        price = _fetch_binance_price(cfg.paper_symbol)
        prev_price = _as_decimal(state.get("paper_last_price", "0"))
        position = state.get("paper_position")
        trades = list(state.get("paper_trades") or [])
        realized = _as_decimal(state.get("paper_realized_pnl_usdc", "0"))

        # Baseline heuristic: momentum filter for entry/exit.
        action = "hold"
        if prev_price > Decimal("0"):
            up = price >= (prev_price * Decimal("1.002"))
            down = price <= (prev_price * Decimal("0.998"))
            if not position and up:
                action = "buy"
            elif position and down:
                action = "sell"

        ai_signal = self._ai_trade_signal(
            cfg,
            symbol=cfg.paper_symbol,
            price=price,
            prev_price=prev_price,
            has_position=bool(position),
        )
        if (
            ai_signal.get("used")
            and ai_signal.get("confidence", 0.0) >= cfg.ai_min_confidence
        ):
            action = str(ai_signal.get("action", action))

        event = "paper_hold"
        now_iso = datetime.now(timezone.utc).isoformat()

        if not position and action == "buy" and cfg.paper_notional_usdc > Decimal("0"):
            qty = (cfg.paper_notional_usdc / price).quantize(
                Decimal("0.00000001"), rounding=ROUND_DOWN
            )
            if qty > Decimal("0"):
                position = {
                    "side": "long",
                    "entry_price": str(price),
                    "qty": str(qty),
                    "opened_at": now_iso,
                }
                event = "paper_open_long"
                trades.append(
                    {
                        "ts": now_iso,
                        "type": "open_long",
                        "price": str(price),
                        "qty": str(qty),
                    }
                )

        elif position and position.get("side") == "long":
            entry = _as_decimal(position.get("entry_price", "0"))
            qty = _as_decimal(position.get("qty", "0"))
            pnl_pct = (
                ((price - entry) / entry) if entry > Decimal("0") else Decimal("0")
            )
            take_profit_hit = pnl_pct >= cfg.paper_take_profit_pct
            stop_loss_hit = pnl_pct <= (Decimal("0") - cfg.paper_stop_loss_pct)
            if action == "sell" or take_profit_hit or stop_loss_hit:
                pnl = ((price - entry) * qty).quantize(
                    Decimal("0.000001"), rounding=ROUND_DOWN
                )
                realized += pnl
                event = (
                    "paper_close_long_tp"
                    if take_profit_hit
                    else "paper_close_long_sl"
                    if stop_loss_hit
                    else "paper_close_long"
                )
                trades.append(
                    {
                        "ts": now_iso,
                        "type": "close_long",
                        "price": str(price),
                        "qty": str(qty),
                        "pnl_usdc": str(pnl),
                    }
                )
                position = None

        state["paper_last_price"] = str(price)
        state["paper_position"] = position
        state["paper_realized_pnl_usdc"] = str(realized)
        state["paper_trades"] = trades[-100:]

        return {
            "event": event,
            "price": str(price),
            "prev_price": str(prev_price),
            "action": action,
            "ai_signal": ai_signal,
            "paper_realized_pnl_usdc": str(realized),
            "paper_position": position,
        }

    def _live_trade_step(
        self,
        cfg: TraderConfig,
        state: Dict[str, Any],
        wallet_tool,
        symbol: str,
        news_items: list[dict[str, Any]],
        balance_snapshot: Optional[Dict[str, Any]] = None,
        wallet_network: str = "",
        wallet_networks: Optional[list[str]] = None,
        pre_news_guard: Optional[Dict[str, Any]] = None,
        rank_score: Optional[Decimal] = None,
        execution_backend_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        asset = _asset_from_symbol(symbol)
        price = _fetch_binance_price(symbol)
        live_last_prices = dict(state.get("live_last_prices") or {})
        live_positions = dict(state.get("live_positions") or {})
        live_realized_map = dict(state.get("live_realized_pnl_usdc_by_symbol") or {})
        live_entry_momentum_map = dict(
            state.get("live_entry_momentum_pct_by_symbol") or {}
        )
        live_exit_momentum_map = dict(
            state.get("live_exit_momentum_pct_by_symbol") or {}
        )
        live_hold_streak_map = dict(state.get("live_hold_streak_by_symbol") or {})
        live_last_action_ts_map = dict(state.get("live_last_action_ts_by_symbol") or {})
        live_price_history_map = dict(state.get("live_price_history_by_symbol") or {})
        first_symbol = _live_symbols(cfg)[0] if _live_symbols(cfg) else symbol
        execution_backend = (
            str(execution_backend_override or cfg.live_execution_backend)
            .strip()
            .lower()
        )
        risk_scope = _live_risk_scope(execution_backend, wallet_network)
        risk_profiles = dict(state.get("live_risk_profiles") or {})
        risk_state = dict(risk_profiles.get(risk_scope) or {})

        # Ensure BNB pairs run against BNB chain balances/swaps instead of
        # whichever network currently has the most USDC.
        if execution_backend == "wallet" and asset in {"BNB", "WBNB"}:
            wallet_network = "bnb"

        # Backward compatibility: migrate legacy single-symbol fields on first read.
        if (
            symbol == first_symbol
            and symbol not in live_last_prices
            and state.get("live_last_price")
        ):
            live_last_prices[symbol] = str(state.get("live_last_price"))
        if (
            symbol == first_symbol
            and symbol not in live_positions
            and state.get("live_position")
        ):
            legacy_pos = state.get("live_position")
            if isinstance(legacy_pos, dict):
                live_positions[symbol] = dict(legacy_pos)
        if (
            symbol == first_symbol
            and symbol not in live_realized_map
            and state.get("live_realized_pnl_usdc") is not None
        ):
            live_realized_map[symbol] = str(state.get("live_realized_pnl_usdc"))

        prev_price = _as_decimal(live_last_prices.get(symbol, "0"))
        position = live_positions.get(symbol)
        foreign_position = False
        if isinstance(position, dict):
            position_backend = str(
                position.get("execution_backend") or "wallet"
            ).strip().lower()
            if position_backend != execution_backend:
                # A different account owns this position.  Never attempt to
                # close it through the current backend or count it against the
                # current account's position cap.
                foreign_position = True
                position = None

        # Reconcile stale wallet positions: if state says we are long but on-chain
        # token balance is already zero, clear the position to avoid permanent holds.
        if execution_backend == "wallet" and isinstance(position, dict):
            try:
                probe_network = _normalize_wallet_network(
                    str(position.get("wallet_network", wallet_network) or "")
                )
                bal_args: Dict[str, Any] = {
                    "token_symbol": _wallet_token_symbol(asset),
                }
                if probe_network:
                    bal_args["network"] = probe_network
                tok_probe = wallet_tool("token_balance", bal_args)
                if tok_probe.get("ok"):
                    live_amt = _as_decimal(tok_probe.get("amount", "0"), "0")
                    if live_amt <= Decimal("0"):
                        position = None
            except Exception:
                pass

        trades = list(state.get("live_trades") or [])
        realized = _as_decimal(live_realized_map.get(symbol, "0"))
        if cfg.live_learning_enabled:
            entry_momentum = _as_decimal(
                live_entry_momentum_map.get(symbol, str(cfg.live_entry_momentum_pct)),
                str(cfg.live_entry_momentum_pct),
            )
            exit_momentum = _as_decimal(
                live_exit_momentum_map.get(symbol, str(cfg.live_exit_momentum_pct)),
                str(cfg.live_exit_momentum_pct),
            )
        else:
            # When learning is off, prefer explicit config over stale symbol state.
            entry_momentum = _as_decimal(
                str(cfg.live_entry_momentum_pct), str(cfg.live_entry_momentum_pct)
            )
            exit_momentum = _as_decimal(
                str(cfg.live_exit_momentum_pct), str(cfg.live_exit_momentum_pct)
            )
            live_entry_momentum_map[symbol] = str(entry_momentum)
            live_exit_momentum_map[symbol] = str(exit_momentum)

        price_history = list(live_price_history_map.get(symbol) or [])
        price_history.append(str(price))
        if len(price_history) > 120:
            price_history = price_history[-120:]
        sma_fast = _sma_decimal(price_history, 5)
        sma_slow = _sma_decimal(price_history, 20)
        pattern_score = _pattern_forecast(price_history)
        pattern_reliability = _pattern_reliability(price_history)
        effective_pattern_score = pattern_score * pattern_reliability
        regime = "flat"
        if sma_fast > Decimal("0") and sma_slow > Decimal("0"):
            if sma_fast >= sma_slow * Decimal("1.0005"):
                regime = "bull"
            elif sma_fast <= sma_slow * Decimal("0.9995"):
                regime = "bear"

        def _wallet(
            action: str, payload: Dict[str, Any] | None = None
        ) -> Dict[str, Any]:
            call_args = dict(payload or {})
            if wallet_network and "network" not in call_args:
                call_args["network"] = wallet_network
            return wallet_tool(action, call_args)

        token_free_amount = Decimal("0")
        wallet_network_balances: Dict[str, Dict[str, str]] = {}
        wallet_network_errors: Dict[str, str] = {}
        wallet_unsupported_networks: list[str] = []
        wallet_total_usdc = Decimal("0")
        if execution_backend == "binance_spot":
            acct = _wallet("binance_account", {"non_zero_only": True})
            if not acct.get("ok"):
                raise RuntimeError(
                    f"binance_account_error:{acct.get('error', 'unknown')}"
                )
            quote_asset = _quote_from_symbol(symbol)
            balances = list(acct.get("balances") or [])
            quote_free = Decimal("0")
            token_free = Decimal("0")
            for row in balances:
                asset_row = str((row or {}).get("asset", "")).upper()
                free_amt = _as_decimal((row or {}).get("free", "0"))
                if asset_row == quote_asset:
                    quote_free = free_amt
                if asset_row == asset:
                    token_free = free_amt
            usdc_amount = quote_free
            token_free_amount = token_free
            native_eth = Decimal("0")
            bal = {"ok": True, "quote_asset": quote_asset, "balances": balances}
        elif execution_backend == "metamask_perps":
            acct = _wallet("metamask_perps_account", {})
            if not acct.get("ok"):
                raise RuntimeError(
                    f"metamask_perps_account_error:{acct.get('error', 'unknown')}"
                )

            account = acct.get("account")
            available = Decimal("0")
            if isinstance(account, dict):
                available = _as_decimal(
                    account.get("available_usdc", account.get("availableBalance", "0")),
                    "0",
                )
            usdc_amount = available
            token_free_amount = Decimal("0")
            native_eth = Decimal("0")
            bal = {"ok": True, "account": account}
        elif execution_backend == "hyperliquid_perps":
            acct = _wallet("hyperliquid_perps_account", {})
            if not acct.get("ok"):
                raise RuntimeError(
                    f"hyperliquid_perps_account_error:{acct.get('error', 'unknown')}"
                )

            account = acct.get("account")
            available = Decimal("0")
            if isinstance(account, dict):
                available = _as_decimal(
                    account.get("available_usdc", account.get("availableBalance", "0")),
                    "0",
                )
            usdc_amount = available
            token_free_amount = Decimal("0")
            native_eth = Decimal("0")
            bal = {"ok": True, "account": account}
        else:
            requested_networks = list(wallet_networks or _live_wallet_networks(cfg))
            explicit_network = _normalize_wallet_network(wallet_network)
            if explicit_network and explicit_network not in requested_networks:
                requested_networks.insert(0, explicit_network)

            selected_wallet_network = ""
            selected_balance: Optional[Dict[str, Any]] = None
            use_multi_network_scan = len(requested_networks) > 1
            wallet_network_results: Dict[str, Dict[str, Any]] = {}

            if use_multi_network_scan:
                for net in requested_networks:
                    net_bal = wallet_tool("balance", {"network": net})
                    if isinstance(net_bal, str):
                        try:
                            net_bal = json.loads(net_bal)
                        except json.JSONDecodeError:
                            wallet_network_errors[net] = (
                                "unexpected_balance_payload:str"
                            )
                            continue
                    if not isinstance(net_bal, dict):
                        wallet_network_errors[net] = (
                            f"unexpected_balance_payload:{type(net_bal).__name__}"
                        )
                        continue
                    if not net_bal.get("ok"):
                        err_text = str(net_bal.get("error", "unknown"))
                        if "unsupported network" in err_text.lower():
                            wallet_unsupported_networks.append(net)
                        else:
                            wallet_network_errors[net] = err_text
                        continue
                    net_usdc = _as_decimal(
                        (net_bal.get("usdc") or {}).get("amount", "0")
                    )
                    net_native = _as_decimal(
                        (net_bal.get("native") or {}).get("eth", "0")
                    )
                    wallet_network_results[net] = net_bal
                    wallet_total_usdc += net_usdc
                    wallet_network_balances[net] = {
                        "chain": str(net_bal.get("chain", net)),
                        "usdc": str(net_usdc),
                        "native": str(net_native),
                    }

                if not wallet_network_balances:
                    if wallet_network_errors:
                        first_error = next(iter(wallet_network_errors.values()))
                        raise RuntimeError(f"balance_error:{first_error}")
                    raise RuntimeError("balance_error:no_network_balance")

                pinned_network = _normalize_wallet_network(
                    str((position or {}).get("wallet_network", ""))
                )
                if pinned_network in wallet_network_balances:
                    selected_wallet_network = pinned_network
                elif explicit_network in wallet_network_balances:
                    selected_wallet_network = explicit_network
                else:
                    selected_wallet_network = max(
                        wallet_network_balances,
                        key=lambda n: _as_decimal(
                            (wallet_network_balances.get(n) or {}).get("usdc", "0")
                        ),
                    )

                selected_balance = wallet_network_results.get(selected_wallet_network)
                if not selected_balance:
                    raise RuntimeError("balance_error:no_selected_network_balance")
                if isinstance(selected_balance, str):
                    try:
                        selected_balance = json.loads(selected_balance)
                    except json.JSONDecodeError:
                        raise RuntimeError(
                            "balance_error:unexpected_selected_balance_payload"
                        )
                if not isinstance(selected_balance, dict):
                    raise RuntimeError(
                        f"balance_error:unexpected_selected_balance_payload:{type(selected_balance).__name__}"
                    )
                bal = selected_balance
                wallet_network = selected_wallet_network
                usdc_amount = _as_decimal((bal.get("usdc") or {}).get("amount", "0"))
                native_eth = _as_decimal((bal.get("native") or {}).get("eth", "0"))
            else:
                if requested_networks:
                    wallet_network = requested_networks[0]
                bal = balance_snapshot or _wallet("balance", {})
                if isinstance(bal, str):
                    try:
                        bal = json.loads(bal)
                    except json.JSONDecodeError:
                        raise RuntimeError(
                            "balance_error:unexpected_balance_payload:str"
                        )
                if not isinstance(bal, dict):
                    raise RuntimeError(
                        f"balance_error:unexpected_balance_payload:{type(bal).__name__}"
                    )
                if not bal.get("ok"):
                    raise RuntimeError(f"balance_error:{bal.get('error', 'unknown')}")
                usdc_amount = _as_decimal((bal.get("usdc") or {}).get("amount", "0"))
                native_eth = _as_decimal((bal.get("native") or {}).get("eth", "0"))
                wallet_total_usdc = usdc_amount
                if wallet_network:
                    wallet_network_balances[wallet_network] = {
                        "chain": str(bal.get("chain", wallet_network)),
                        "usdc": str(usdc_amount),
                        "native": str(native_eth),
                    }

            state["live_wallet_network_balances"] = wallet_network_balances
            state["live_wallet_network_errors"] = wallet_network_errors
            state["live_wallet_unsupported_networks"] = wallet_unsupported_networks
            state["live_wallet_total_usdc"] = str(wallet_total_usdc)
            state["live_wallet_selected_network"] = wallet_network
        # Use the already imported datetime type here.  Some long-lived plugin
        # reload paths rebuild the agent module globals and have been observed
        # to omit the plain ``time`` module, which must never stop a live tick.
        now_ts = int(datetime.now(timezone.utc).timestamp())
        inventory_reconcile_note = ""

        def _token_ref_pair(tok: str) -> str:
            t = str(tok or "").upper()
            if not t:
                return ""
            if t in {"ETH", "WETH"}:
                return "ETHUSDT"
            if t in {"BTC", "WBTC", "CBTC", "BTCB"}:
                return "BTCUSDT"
            if t in {"BNB", "WBNB"}:
                return "BNBUSDT"
            if t in {"MATIC", "POL"}:
                return "MATICUSDT"
            return f"{t}USDT"

        def _try_swap_token_to_usdc(
            token_symbol: str,
            needed_usdc: Decimal,
            *,
            networks: Optional[list[str]] = None,
        ) -> tuple[Dict[str, Any] | None, Decimal, str]:
            token = str(token_symbol or "").upper()
            if not token or token in {"USDC", "NATIVE", "ETH", "WETH"}:
                return None, Decimal("0"), ""

            pair = _token_ref_pair(token)
            if not pair:
                return None, Decimal("0"), ""

            try:
                ref_price = _fetch_binance_price(pair)
            except Exception:
                ref_price = Decimal("0")
            if ref_price <= Decimal("0"):
                return None, Decimal("0"), ""

            search_networks = list(
                networks or ([] if not wallet_network else [wallet_network])
            )
            if not search_networks:
                search_networks = [""]

            tok_amount = Decimal("0")
            chosen_network = ""
            for network_name in search_networks:
                bal_args: Dict[str, Any] = {"token_symbol": token}
                if network_name:
                    bal_args["network"] = network_name
                try:
                    probe = wallet_tool("token_balance", bal_args)
                except Exception:
                    logger.debug(
                        "Optional token balance probe failed for %s on %s",
                        token,
                        network_name or "default",
                        exc_info=True,
                    )
                    continue
                probe_amount = _as_decimal((probe or {}).get("amount", "0"), "0")
                if probe_amount > tok_amount:
                    tok_amount = probe_amount
                    chosen_network = network_name

            if tok_amount <= Decimal("0"):
                return None, Decimal("0"), ""

            sell_amount = min(
                tok_amount,
                (needed_usdc / ref_price * Decimal("1.05")).quantize(
                    Decimal("0.000000000000000001"),
                    rounding=ROUND_DOWN,
                ),
            )
            if sell_amount <= Decimal("0"):
                return None, Decimal("0"), ""

            swap_args: Dict[str, Any] = {
                "token_symbol": token,
                "amount_token": str(sell_amount),
                "slippage_pct": cfg.live_slippage_pct,
            }
            if cfg.confirm_code:
                swap_args["confirm_code"] = cfg.confirm_code
            if chosen_network:
                swap_args["network"] = chosen_network
            return _wallet("swap_token_to_usdc", swap_args), sell_amount, chosen_network

        def _best_native_sellable_eth(
            *,
            networks: Optional[list[str]] = None,
        ) -> tuple[Decimal, str]:
            search_networks = list(
                networks or ([] if not wallet_network else [wallet_network])
            )
            if not search_networks:
                search_networks = [""]
            best_sell = Decimal("0")
            best_network = ""
            for network_name in search_networks:
                bal_args: Dict[str, Any] = {}
                if network_name:
                    bal_args["network"] = network_name
                try:
                    bal = wallet_tool("balance", bal_args)
                except Exception:
                    logger.debug(
                        "Optional native balance probe failed on %s",
                        network_name or "default",
                        exc_info=True,
                    )
                    continue
                if not bal.get("ok"):
                    continue
                native_amt = _as_decimal((bal.get("native") or {}).get("eth", "0"), "0")
                sell_eth = max(Decimal("0"), native_amt - cfg.min_gas_eth)
                if sell_eth > best_sell:
                    best_sell = sell_eth
                    best_network = network_name
            return best_sell, best_network

        def _best_weth_balance(
            *,
            networks: Optional[list[str]] = None,
        ) -> tuple[Decimal, str]:
            search_networks = list(
                networks or ([] if not wallet_network else [wallet_network])
            )
            if not search_networks:
                search_networks = [""]
            best_amt = Decimal("0")
            best_network = ""
            for network_name in search_networks:
                bal_args: Dict[str, Any] = {"token_symbol": "WETH"}
                if network_name:
                    bal_args["network"] = network_name
                try:
                    probe = wallet_tool("token_balance", bal_args)
                except Exception:
                    logger.debug(
                        "Optional WETH balance probe failed on %s",
                        network_name or "default",
                        exc_info=True,
                    )
                    continue
                probe_amt = _as_decimal((probe or {}).get("amount", "0"), "0")
                if probe_amt > best_amt:
                    best_amt = probe_amt
                    best_network = network_name
            return best_amt, best_network

        # If there are no tracked live positions but tradable inventory remains
        # in WETH, rebalance part of it back to USDC so the next entry is fundable.
        if (
            execution_backend == "wallet"
            and symbol == first_symbol
            and not live_positions
            and cfg.live_notional_usdc >= cfg.live_min_trade_usdc
        ):
            reconcile_last_ts = int(
                state.get("live_inventory_reconcile_last_ts", 0) or 0
            )
            reconcile_cooldown_sec = 300
            if (now_ts - reconcile_last_ts) >= reconcile_cooldown_sec:
                available_usdc = max(Decimal("0"), usdc_amount - cfg.reserve_usdc)
                target_usdc = max(cfg.live_min_trade_usdc, cfg.live_notional_usdc)
                needed_usdc = max(Decimal("0"), target_usdc - available_usdc)
                if needed_usdc > Decimal("0"):
                    eth_ref_price = Decimal("0")
                    try:
                        eth_ref_price = _fetch_binance_price("ETHUSDT")
                    except Exception:
                        eth_ref_price = Decimal("0")

                    if eth_ref_price > Decimal("0"):
                        rebalance_res: Dict[str, Any] | None = None
                        rebalance_network = ""
                        sold_weth = Decimal("0")
                        sold_eth = Decimal("0")
                        weth_amount, weth_network = _best_weth_balance(
                            networks=wallet_networks
                        )
                        if weth_amount > Decimal("0"):
                            sold_weth = min(
                                weth_amount,
                                (
                                    needed_usdc / eth_ref_price * Decimal("1.03")
                                ).quantize(
                                    Decimal("0.000000000000000001"),
                                    rounding=ROUND_DOWN,
                                ),
                            )
                            if sold_weth > Decimal("0"):
                                swap_args: Dict[str, Any] = {
                                    "token_symbol": "WETH",
                                    "amount_token": str(sold_weth),
                                    "slippage_pct": cfg.live_slippage_pct,
                                }
                                if cfg.confirm_code:
                                    swap_args["confirm_code"] = cfg.confirm_code
                                if weth_network:
                                    swap_args["network"] = weth_network
                                rebalance_res = _wallet("swap_token_to_usdc", swap_args)
                                rebalance_network = weth_network

                        if rebalance_res is None or not rebalance_res.get("ok"):
                            sellable_eth, sell_network = _best_native_sellable_eth(
                                networks=wallet_networks
                            )
                            sold_eth = sellable_eth
                            if sold_eth > Decimal("0"):
                                sold_eth = min(
                                    sold_eth,
                                    (
                                        needed_usdc / eth_ref_price * Decimal("1.03")
                                    ).quantize(
                                        Decimal("0.000000000000000001"),
                                        rounding=ROUND_DOWN,
                                    ),
                                )
                            if sold_eth > Decimal("0"):
                                swap_args = {
                                    "amount_eth": str(sold_eth),
                                    "slippage_pct": cfg.live_slippage_pct,
                                }
                                if cfg.confirm_code:
                                    swap_args["confirm_code"] = cfg.confirm_code
                                if sell_network:
                                    swap_args["network"] = sell_network
                                rebalance_res = _wallet("swap_eth_to_usdc", swap_args)
                                rebalance_network = sell_network

                        if rebalance_res is None or not rebalance_res.get("ok"):
                            watch_candidates = [
                                t
                                for t in _wallet_watch_symbols(cfg)
                                if t not in {"USDC", "NATIVE", "ETH", "WETH"}
                            ]
                            for tok in watch_candidates:
                                rebalance_res, sold_generic, sold_network = (
                                    _try_swap_token_to_usdc(
                                        tok,
                                        needed_usdc,
                                        networks=wallet_networks,
                                    )
                                )
                                if sold_generic > Decimal("0"):
                                    inventory_reconcile_note = (
                                        "inventory_reconcile_usdc: "
                                        f"sold_token={tok}:{sold_generic} "
                                        f"network={sold_network or wallet_network} "
                                        f"target={target_usdc}"
                                    )
                                if rebalance_res is not None and rebalance_res.get(
                                    "ok"
                                ):
                                    break

                        if rebalance_res is not None:
                            state["live_inventory_reconcile_last_ts"] = now_ts
                            if rebalance_res.get("ok"):
                                balance_network = rebalance_network or wallet_network
                                rebalance_balance = _wallet(
                                    "balance",
                                    {"network": balance_network}
                                    if balance_network
                                    else {},
                                )
                                if rebalance_balance.get("ok"):
                                    usdc_amount = _as_decimal(
                                        (rebalance_balance.get("usdc") or {}).get(
                                            "amount", "0"
                                        )
                                    )
                                    native_eth = _as_decimal(
                                        (rebalance_balance.get("native") or {}).get(
                                            "eth", "0"
                                        )
                                    )
                                    if wallet_network:
                                        wallet_network_balances[wallet_network] = {
                                            "chain": str(
                                                rebalance_balance.get(
                                                    "chain", wallet_network
                                                )
                                            ),
                                            "usdc": str(usdc_amount),
                                            "native": str(native_eth),
                                        }
                                    state["live_wallet_total_usdc"] = str(usdc_amount)
                                    inventory_reconcile_note = (
                                        "inventory_reconcile_usdc: "
                                        f"sold_weth={sold_weth} sold_eth={sold_eth} "
                                        f"target={target_usdc}"
                                    )
                            else:
                                err = str(rebalance_res.get("error", "unknown"))
                                inventory_reconcile_note = (
                                    f"inventory_reconcile_failed: {err}"
                                )

        # Dynamic growth sizing: use total deployable equity (free USDC + capital already in open positions).
        deployed_usdc = Decimal("0")
        for pos in live_positions.values():
            if isinstance(pos, dict) and str(
                pos.get("execution_backend") or "wallet"
            ).strip().lower() == execution_backend:
                deployed_usdc += max(Decimal("0"), _as_decimal(pos.get("usdc_in", "0")))

        equity_usdc = max(Decimal("0"), usdc_amount + deployed_usdc - cfg.reserve_usdc)
        deployable_baseline = max(
            Decimal("0"), cfg.base_capital_usdc - cfg.reserve_usdc
        )
        growth_scale = Decimal("1")
        effective_notional = cfg.live_notional_usdc
        if cfg.growth_enabled:
            if deployable_baseline > Decimal("0"):
                growth_gain = max(Decimal("0"), equity_usdc - deployable_baseline)
                growth_scale += (
                    growth_gain / deployable_baseline
                ) * cfg.growth_reinvest_ratio
            effective_notional = cfg.live_notional_usdc * growth_scale
            if cfg.growth_max_notional_usdc > Decimal("0"):
                effective_notional = min(
                    effective_notional, cfg.growth_max_notional_usdc
                )
            effective_notional = effective_notional.quantize(
                Decimal("0.000001"), rounding=ROUND_DOWN
            )

        # Daily risk state and brakes.
        risk_day = str(risk_state.get("day") or "")
        today = _today_utc()
        if risk_day != today:
            risk_state["day"] = today
            risk_state["day_start_equity_usdc"] = str(equity_usdc)
            risk_state["day_peak_equity_usdc"] = str(equity_usdc)
            risk_state["stop_until"] = 0
            risk_state["consecutive_losses"] = 0

        risk_peak = _as_decimal(risk_state.get("day_peak_equity_usdc", "0"))
        if risk_peak <= Decimal("0"):
            risk_peak = equity_usdc
        # Self-heal corrupted peak-equity anchor. A stored peak that is wildly
        # larger than current equity (e.g. from a transient mis-scaled balance
        # read such as an 18-decimal token reported as 6-decimal) would lock the
        # daily drawdown brake at ~100% forever. Detect that and reset the daily
        # risk anchor to the real equity so the agent can trade again.
        if equity_usdc > Decimal("0") and risk_peak > equity_usdc * Decimal("50"):
            risk_peak = equity_usdc
            risk_state["day_start_equity_usdc"] = str(equity_usdc)
            risk_state["stop_until"] = 0
            risk_state["consecutive_losses"] = 0
        risk_peak = max(risk_peak, equity_usdc)
        risk_state["day_peak_equity_usdc"] = str(risk_peak)

        risk_drawdown_pct = Decimal("0")
        if risk_peak > Decimal("0"):
            risk_drawdown_pct = (risk_peak - equity_usdc) / risk_peak

        # Fail-open for transient invalid equity snapshots to avoid false
        # full-drawdown locks in live mode.
        if equity_usdc <= Decimal("0"):
            risk_drawdown_pct = Decimal("0")

        risk_stop_until = int(risk_state.get("stop_until", 0) or 0)
        consecutive_losses = int(risk_state.get("consecutive_losses", 0) or 0)
        if risk_stop_until > 0 and now_ts >= risk_stop_until:
            risk_stop_until = 0
            consecutive_losses = 0
            risk_state["stop_until"] = 0
            risk_state["consecutive_losses"] = 0
        if equity_usdc <= Decimal("0"):
            risk_stop_until = 0
            consecutive_losses = 0
            risk_state["stop_until"] = 0
            risk_state["consecutive_losses"] = 0
        cooldown_until = now_ts + (cfg.live_risk_cooldown_minutes * 60)
        if (
            risk_drawdown_pct >= cfg.live_risk_daily_max_drawdown_pct
            and risk_stop_until <= 0
        ):
            risk_stop_until = cooldown_until
            risk_state["stop_until"] = risk_stop_until
        if (
            consecutive_losses >= cfg.live_risk_max_consecutive_losses
            and risk_stop_until <= 0
        ):
            risk_stop_until = cooldown_until
            risk_state["stop_until"] = risk_stop_until
        risk_brake_active = now_ts < risk_stop_until

        # Baseline heuristic: momentum filter for entry/exit.
        action = "hold"
        if prev_price > Decimal("0"):
            up = price >= (prev_price * (Decimal("1") + entry_momentum))
            down = price <= (prev_price * (Decimal("1") - entry_momentum))
            if not position:
                if up:
                    action = "buy"
                elif down:
                    action = "short"
            elif position:
                # Cierre de posición si la tendencia se revierte
                if (position.get("side") == "long" and down) or (
                    position.get("side") == "short" and up
                ):
                    action = "sell"

        # The last tick is noisy. A sufficiently strong multi-horizon pattern
        # may initiate or close a position before the one-tick trigger fires.
        if not position and effective_pattern_score >= Decimal("0.22"):
            action = "buy"
        elif not position and effective_pattern_score <= Decimal("-0.22"):
            action = "short"
        elif position and (
            (
                position.get("side") == "long"
                and effective_pattern_score <= Decimal("-0.30")
            )
            or (
                position.get("side") == "short"
                and effective_pattern_score >= Decimal("0.30")
            )
        ):
            action = "sell"

        news_guard = pre_news_guard or self._news_guard_for_symbol(
            cfg, symbol=symbol, news_items=news_items
        )

        ai_signal = self._ai_trade_signal(
            cfg,
            symbol=symbol,
            price=price,
            prev_price=prev_price,
            has_position=bool(position),
            news_context=(
                f"score={news_guard.get('score', 0)}; "
                f"relevant={news_guard.get('relevant_items', 0)}; "
                f"headlines={news_guard.get('summary', 'none')}"
            ),
        )
        if (
            ai_signal.get("used")
            and ai_signal.get("confidence", 0.0) >= cfg.ai_min_confidence
        ):
            action = str(ai_signal.get("action", action))

        opportunity_score = rank_score
        if opportunity_score is None:
            momentum_pct = (
                ((price - prev_price) / prev_price)
                if prev_price > Decimal("0")
                else Decimal("0")
            )
            regime_score = (
                Decimal("1")
                if regime == "bull"
                else Decimal("-1")
                if regime == "bear"
                else Decimal("0")
            )
            news_score = Decimal(str(int(news_guard.get("score") or 0))) / Decimal("4")
            position_bonus = cfg.live_score_position_bonus if position else Decimal("0")
            opportunity_score = (
                momentum_pct * cfg.live_score_momentum_weight
                + news_score * cfg.live_score_news_weight
                + regime_score * cfg.live_score_regime_weight
                + position_bonus
            )

        current_hold_streak = int(live_hold_streak_map.get(symbol, 0) or 0)
        adaptive_min_opportunity_score = _adaptive_opportunity_floor(
            base_score=cfg.live_min_opportunity_score,
            hold_streak=current_hold_streak if not position else 0,
            relax_after_holds=cfg.live_opportunity_relax_after_holds,
            relax_step=cfg.live_opportunity_relax_step_pct,
            min_actions_per_hour=cfg.live_min_actions_per_hour,
        )

        if not position and opportunity_score < adaptive_min_opportunity_score:
            action = "hold"

        # Opportunistic fallback: if score is acceptable and price is rising,
        # allow a buy attempt even when strict momentum threshold is not met.
        if (
            not position
            and action == "hold"
            and opportunity_score >= adaptive_min_opportunity_score
            and prev_price > Decimal("0")
            and price > prev_price
        ):
            action = "buy"

        # Regime filter: avoid opening longs in bear regime unless AI confidence is exceptional.
        ai_conf = float(ai_signal.get("confidence", 0.0) or 0.0)
        if (
            not position
            and action == "buy"
            and regime == "bear"
            and ai_conf < 0.9
            and cfg.live_score_regime_weight > Decimal("0")
        ):
            action = "hold"

        # If already in position and regime turns clearly bearish, bias to risk-off.
        if (
            position
            and action == "hold"
            and regime == "bear"
            and prev_price > Decimal("0")
            and price < prev_price
        ):
            action = "sell"

        wanted_buy = bool((not position) and action in ("buy", "short"))
        position_cap_blocked = bool(
            wanted_buy
            and (
                foreign_position
                or len(
                    [
                        p
                        for p in live_positions.values()
                        if isinstance(p, dict)
                        and str(p.get("execution_backend") or "wallet")
                        .strip()
                        .lower()
                        == execution_backend
                    ]
                )
                >= cfg.live_max_open_positions
            )
        )
        if position_cap_blocked:
            action = "hold"
            wanted_buy = False
        if risk_brake_active and wanted_buy:
            action = "hold"
            wanted_buy = False
        if not position and action == "buy" and news_guard.get("block_buy"):
            action = "hold"

        event = "live_hold"
        now_iso = datetime.now(timezone.utc).isoformat()
        tx_hash = ""
        live_error = ""
        hold_reason = ""
        if position_cap_blocked:
            hold_reason = "max_open_positions"
        learning_note = inventory_reconcile_note
        close_pnl_pct = Decimal("0")

        if not position and action in ("buy", "short"):
            if execution_backend in {"metamask_perps", "hyperliquid_perps"}:
                # The wallet reserve belongs to the on-chain wallet.  Applying
                # it again to an isolated perps balance can incorrectly reduce
                # usable margin to zero and suppress every valid order.
                available_usdc = max(Decimal("0"), usdc_amount)
            else:
                available_usdc = max(Decimal("0"), usdc_amount - cfg.reserve_usdc)
            max_trade_usdc = available_usdc
            if execution_backend in {"metamask_perps", "hyperliquid_perps"}:
                leverage_cap = Decimal(
                    str(max(1, min(100, int(cfg.live_perps_leverage))))
                )
                max_trade_usdc = available_usdc * leverage_cap
            buy_usdc = min(effective_notional, max_trade_usdc)
            if (
                execution_backend in {"metamask_perps", "hyperliquid_perps"}
                and buy_usdc < cfg.live_min_trade_usdc
                and max_trade_usdc >= cfg.live_min_trade_usdc
                and effective_notional > Decimal("0")
            ):
                buy_usdc = cfg.live_min_trade_usdc
            if (
                execution_backend == "wallet"
                and buy_usdc < cfg.live_min_trade_usdc
                and effective_notional > Decimal("0")
                and asset != "ETH"
            ):
                target_buy_usdc = min(effective_notional, cfg.live_min_trade_usdc)
                needed_usdc = max(Decimal("0"), target_buy_usdc - buy_usdc)
                if needed_usdc > Decimal("0"):
                    auto_fund_error = ""
                    auto_fund_network = ""
                    eth_ref_price = Decimal("0")
                    try:
                        eth_ref_price = _fetch_binance_price("ETHUSDT")
                    except Exception:
                        eth_ref_price = Decimal("0")

                    auto_fund_res: Dict[str, Any] | None = None
                    if eth_ref_price > Decimal("0"):
                        sellable_eth, sell_network = _best_native_sellable_eth(
                            networks=wallet_networks
                        )
                        sell_eth = sellable_eth
                        if sell_eth > Decimal("0"):
                            sell_eth = min(
                                sell_eth,
                                (
                                    needed_usdc / eth_ref_price * Decimal("1.03")
                                ).quantize(
                                    Decimal("0.000000000000000001"),
                                    rounding=ROUND_DOWN,
                                ),
                            )
                        if sell_eth > Decimal("0"):
                            swap_args: Dict[str, Any] = {
                                "amount_eth": str(sell_eth),
                                "slippage_pct": cfg.live_slippage_pct,
                            }
                            if cfg.confirm_code:
                                swap_args["confirm_code"] = cfg.confirm_code
                            if sell_network:
                                swap_args["network"] = sell_network
                            auto_fund_res = _wallet("swap_eth_to_usdc", swap_args)
                            auto_fund_network = sell_network

                    if auto_fund_res is None or not auto_fund_res.get("ok"):
                        weth_amount, weth_network = _best_weth_balance(
                            networks=wallet_networks
                        )
                        if weth_amount > Decimal("0") and eth_ref_price > Decimal("0"):
                            sell_weth = min(
                                weth_amount,
                                (
                                    needed_usdc / eth_ref_price * Decimal("1.03")
                                ).quantize(
                                    Decimal("0.000000000000000001"),
                                    rounding=ROUND_DOWN,
                                ),
                            )
                            if sell_weth > Decimal("0"):
                                swap_args = {
                                    "token_symbol": "WETH",
                                    "amount_token": str(sell_weth),
                                    "slippage_pct": cfg.live_slippage_pct,
                                }
                                if cfg.confirm_code:
                                    swap_args["confirm_code"] = cfg.confirm_code
                                if weth_network:
                                    swap_args["network"] = weth_network
                                auto_fund_res = _wallet("swap_token_to_usdc", swap_args)
                                auto_fund_network = weth_network

                    if auto_fund_res is None or not auto_fund_res.get("ok"):
                        watch_candidates = [
                            t
                            for t in _wallet_watch_symbols(cfg)
                            if t not in {"USDC", "NATIVE", "ETH", "WETH", asset}
                        ]
                        for tok in watch_candidates:
                            auto_fund_res, sold_generic, sold_network = (
                                _try_swap_token_to_usdc(
                                    tok,
                                    needed_usdc,
                                    networks=wallet_networks,
                                )
                            )
                            if auto_fund_res and auto_fund_res.get("ok"):
                                auto_fund_network = sold_network
                                learning_note = (
                                    f"auto_fund_usdc_token: token={tok} sold={sold_generic} "
                                    f"network={sold_network or wallet_network} "
                                    f"needed={needed_usdc}"
                                )
                                break

                    if auto_fund_res and auto_fund_res.get("ok"):
                        balance_network = auto_fund_network or wallet_network
                        rebalance_balance = _wallet(
                            "balance",
                            {"network": balance_network} if balance_network else {},
                        )
                        if rebalance_balance.get("ok"):
                            usdc_amount = _as_decimal(
                                (rebalance_balance.get("usdc") or {}).get("amount", "0")
                            )
                            native_eth = _as_decimal(
                                (rebalance_balance.get("native") or {}).get("eth", "0")
                            )
                            available_usdc = max(
                                Decimal("0"), usdc_amount - cfg.reserve_usdc
                            )
                            buy_usdc = min(effective_notional, available_usdc)
                            learning_note = f"auto_fund_usdc: available={available_usdc} needed={needed_usdc}"
                    else:
                        auto_fund_error = str((auto_fund_res or {}).get("error", ""))
                        if auto_fund_error:
                            learning_note = f"auto_fund_failed: {auto_fund_error}"
            if buy_usdc >= cfg.live_min_trade_usdc:
                momentum_edge_pct = (
                    ((price - prev_price) / prev_price)
                    if prev_price > Decimal("0")
                    else Decimal("0")
                )
                # Project expected edge over short trend windows instead of
                # relying only on a single tick delta.
                edge_candidates = [momentum_edge_pct]
                if len(price_history) >= 5:
                    base_5 = _as_decimal(price_history[-5], "0")
                    if base_5 > Decimal("0"):
                        edge_candidates.append((price - base_5) / base_5)
                if len(price_history) >= 15:
                    base_15 = _as_decimal(price_history[-15], "0")
                    if base_15 > Decimal("0"):
                        edge_candidates.append((price - base_15) / base_15)
                if len(price_history) >= 30:
                    base_30 = _as_decimal(price_history[-30], "0")
                    if base_30 > Decimal("0"):
                        edge_candidates.append((price - base_30) / base_30)
                if len(price_history) >= 60:
                    base_60 = _as_decimal(price_history[-60], "0")
                    if base_60 > Decimal("0"):
                        edge_candidates.append((price - base_60) / base_60)
                if len(price_history) >= 90:
                    base_90 = _as_decimal(price_history[-90], "0")
                    if base_90 > Decimal("0"):
                        edge_candidates.append((price - base_90) / base_90)
                if len(price_history) >= 120:
                    base_120 = _as_decimal(price_history[-120], "0")
                    if base_120 > Decimal("0"):
                        edge_candidates.append((price - base_120) / base_120)
                projected_edge_pct = max(edge_candidates)
                cost_native_eth = native_eth
                if execution_backend in {"metamask_perps", "hyperliquid_perps"}:
                    # Perps execution does not consume on-chain gas for each open/close.
                    cost_native_eth = cfg.gas_warning_eth
                roundtrip_cost_usdc = self._estimate_live_roundtrip_cost_usdc(
                    cfg,
                    notional_usdc=buy_usdc,
                    native_eth=cost_native_eth,
                )
                roundtrip_cost_pct = (
                    (roundtrip_cost_usdc / buy_usdc)
                    if buy_usdc > Decimal("0")
                    else Decimal("0")
                )
                adaptive_micro_min_edge_pct = cfg.live_micro_min_net_edge_pct
                last_action_ts_for_micro = int(
                    live_last_action_ts_map.get(symbol, 0) or 0
                )
                micro_relax_steps = _micro_edge_relax_steps(
                    hold_streak=current_hold_streak if not position else 0,
                    relax_after_holds=cfg.live_micro_relax_after_holds,
                    min_actions_per_hour=cfg.live_min_actions_per_hour,
                    last_action_ts=last_action_ts_for_micro,
                    now_ts=now_ts,
                )
                if not position and cfg.live_micro_relax_step_pct > Decimal("0"):
                    if micro_relax_steps > 0:
                        relax_delta = cfg.live_micro_relax_step_pct * Decimal(
                            str(micro_relax_steps)
                        )
                        adaptive_micro_min_edge_pct = max(
                            cfg.live_micro_min_floor_pct,
                            cfg.live_micro_min_net_edge_pct - relax_delta,
                        )

                required_edge_pct = adaptive_micro_min_edge_pct + roundtrip_cost_pct
                if projected_edge_pct < required_edge_pct:
                    event = "live_micro_blocked"
                    hold_reason = "micro_edge_below_required"
                    learning_note = (
                        f"micro_guard: edge={projected_edge_pct} req={required_edge_pct} "
                        f"req_base={adaptive_micro_min_edge_pct} "
                        f"edge_1t={momentum_edge_pct} "
                        f"cost_usdc={roundtrip_cost_usdc}"
                    )
                else:
                    if execution_backend == "binance_spot":
                        order_args: Dict[str, Any] = {
                            "symbol": symbol,
                            "side": "BUY",
                            "type": "MARKET",
                            "quoteOrderQty": str(
                                buy_usdc.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
                            ),
                            "test": cfg.live_binance_test_order,
                        }
                        if cfg.confirm_code:
                            order_args["confirm_code"] = cfg.confirm_code
                        swap_res = _wallet("binance_order", order_args)
                        if swap_res.get("ok"):
                            order = dict(swap_res.get("order") or {})
                            if cfg.live_binance_test_order:
                                event = "live_open_test"
                                tx_hash = str(order.get("orderId", ""))
                                learning_note = (
                                    f"binance_test_order: buy {symbol} quote={buy_usdc}"
                                )
                            else:
                                filled_qty = _as_decimal(order.get("executedQty", "0"))
                                if filled_qty <= Decimal("0") and price > Decimal("0"):
                                    filled_qty = (buy_usdc / price).quantize(
                                        Decimal("0.00000001"),
                                        rounding=ROUND_DOWN,
                                    )
                                position = {
                                    "side": "long",
                                    "entry_price": str(price),
                                    "usdc_in": str(buy_usdc),
                                    "token_symbol": asset,
                                    "token_amount": str(filled_qty),
                                    "opened_at": now_iso,
                                    "execution_backend": "binance_spot",
                                    "binance_symbol": symbol,
                                }
                                event = "live_open_long"
                                tx_hash = str(order.get("orderId", ""))
                                trades.append(
                                    {
                                        "ts": now_iso,
                                        "type": "open_long",
                                        "symbol": symbol,
                                        "price": str(price),
                                        "usdc_in": str(buy_usdc),
                                        "tx_hash": tx_hash,
                                    }
                                )
                        else:
                            event = "live_open_failed"
                            live_error = str(swap_res.get("error", "unknown"))
                            hold_reason = "open_failed"
                    elif execution_backend == "metamask_perps":
                        perps_side = "BUY" if action == "buy" else "SELL"
                        perps_args: Dict[str, Any] = {
                            "symbol": symbol,
                            "side": perps_side,
                            "type": "MARKET",
                            "notional_usdc": str(
                                buy_usdc.quantize(
                                    Decimal("0.000001"), rounding=ROUND_DOWN
                                )
                            ),
                            "leverage": max(1, min(100, int(cfg.live_perps_leverage))),
                        }
                        if cfg.confirm_code:
                            perps_args["confirm_code"] = cfg.confirm_code
                        swap_res = _wallet("metamask_perps_order", perps_args)
                        if swap_res.get("ok"):
                            order = dict(swap_res.get("order") or {})
                            position = {
                                "side": "long" if action == "buy" else "short",
                                "entry_price": str(price),
                                "usdc_in": str(buy_usdc),
                                "token_symbol": asset,
                                "token_amount": "0",
                                "opened_at": now_iso,
                                "execution_backend": "metamask_perps",
                                "perps_symbol": symbol,
                                "perps_leverage": str(
                                    max(1, min(100, int(cfg.live_perps_leverage)))
                                ),
                            }
                            event = (
                                "live_open_long"
                                if action == "buy"
                                else "live_open_short"
                            )
                            tx_hash = str(
                                order.get("order_id")
                                or order.get("id")
                                or order.get("client_order_id")
                                or ""
                            )
                            trades.append(
                                {
                                    "ts": now_iso,
                                    "type": "open_long"
                                    if action == "buy"
                                    else "open_short",
                                    "symbol": symbol,
                                    "price": str(price),
                                    "usdc_in": str(buy_usdc),
                                    "tx_hash": tx_hash,
                                }
                            )
                        else:
                            event = "live_open_failed"
                            live_error = str(swap_res.get("error", "unknown"))
                            hold_reason = "open_failed"
                    elif execution_backend == "hyperliquid_perps":
                        perps_side = "BUY" if action == "buy" else "SELL"
                        perps_args = {
                            "symbol": symbol,
                            "side": perps_side,
                            "type": "MARKET",
                            "notional_usdc": str(
                                buy_usdc.quantize(
                                    Decimal("0.000001"), rounding=ROUND_DOWN
                                )
                            ),
                            "leverage": max(1, min(100, int(cfg.live_perps_leverage))),
                        }
                        if cfg.confirm_code:
                            perps_args["confirm_code"] = cfg.confirm_code
                        swap_res = _wallet("hyperliquid_perps_order", perps_args)
                        if swap_res.get("ok"):
                            order = dict(swap_res.get("order") or {})
                            position = {
                                "side": "long" if action == "buy" else "short",
                                "entry_price": str(price),
                                "usdc_in": str(buy_usdc),
                                "token_symbol": asset,
                                "token_amount": "0",
                                "opened_at": now_iso,
                                "execution_backend": "hyperliquid_perps",
                                "perps_symbol": symbol,
                                "perps_leverage": str(
                                    max(1, min(100, int(cfg.live_perps_leverage)))
                                ),
                            }
                            event = (
                                "live_open_long"
                                if action == "buy"
                                else "live_open_short"
                            )
                            tx_hash = str(
                                order.get("order_id")
                                or order.get("id")
                                or order.get("client_order_id")
                                or ""
                            )
                            trades.append(
                                {
                                    "ts": now_iso,
                                    "type": "open_long"
                                    if action == "buy"
                                    else "open_short",
                                    "symbol": symbol,
                                    "price": str(price),
                                    "usdc_in": str(buy_usdc),
                                    "tx_hash": tx_hash,
                                }
                            )
                        else:
                            event = "live_open_failed"
                            live_error = str(swap_res.get("error", "unknown"))
                            hold_reason = "open_failed"
                    else:
                        position_network = _normalize_wallet_network(wallet_network)
                        swap_args = {
                            "amount_usdc": str(
                                buy_usdc.quantize(
                                    Decimal("0.000001"), rounding=ROUND_DOWN
                                )
                            ),
                            "slippage_pct": cfg.live_slippage_pct,
                        }
                        if cfg.confirm_code:
                            swap_args["confirm_code"] = cfg.confirm_code
                        if asset == "ETH":
                            swap_res = _wallet("swap_usdc_to_eth", swap_args)
                        else:
                            swap_args["token_symbol"] = _wallet_token_symbol(asset)
                            swap_res = _wallet("swap_usdc_to_token", swap_args)
                        if swap_res.get("ok"):
                            position = {
                                "side": "long",
                                "entry_price": str(price),
                                "usdc_in": str(buy_usdc),
                                "token_symbol": asset,
                                "token_amount": str(
                                    _as_decimal(swap_res.get("token_received", "0"))
                                ),
                                "opened_at": now_iso,
                                "wallet_network": position_network,
                            }
                            event = "live_open_long"
                            tx_hash = str(swap_res.get("swap_hash", ""))
                            trades.append(
                                {
                                    "ts": now_iso,
                                    "type": "open_long",
                                    "symbol": symbol,
                                    "price": str(price),
                                    "usdc_in": str(buy_usdc),
                                    "tx_hash": tx_hash,
                                }
                            )
                        else:
                            event = "live_open_failed"
                            live_error = str(swap_res.get("error", "unknown"))
                            hold_reason = "open_failed"
            else:
                hold_reason = "buy_usdc_below_min_trade"
        elif not position and wanted_buy and news_guard.get("block_buy"):
            event = "live_news_blocked"
            hold_reason = "news_block_buy"
        elif not position and risk_brake_active:
            event = "live_risk_brake"
            hold_reason = "risk_brake_active"

        elif position and position.get("side") == "long":
            if execution_backend == "wallet":
                try:
                    usdc_in_probe = _as_decimal(position.get("usdc_in", "0"), "0")
                    usdc_restored = (
                        usdc_amount >= (usdc_in_probe * Decimal("0.98"))
                        if usdc_in_probe > Decimal("0")
                        else False
                    )
                    bal_args: Dict[str, Any] = {
                        "token_symbol": _wallet_token_symbol(asset),
                    }
                    if wallet_network:
                        bal_args["network"] = wallet_network
                    live_tok = wallet_tool("token_balance", bal_args)
                    live_amt = _as_decimal((live_tok or {}).get("amount", "0"), "0")
                    if usdc_restored and (
                        not live_tok.get("ok") or live_amt <= Decimal("0")
                    ):
                        event = "live_close_reconciled"
                        hold_reason = ""
                        live_error = ""
                        position = None
                except Exception:
                    pass

            if position is None:
                live_positions.pop(symbol, None)
                position = {}

            entry = _as_decimal(position.get("entry_price", "0"))
            usdc_in = _as_decimal(position.get("usdc_in", "0"))
            pnl_pct = (
                ((price - entry) / entry) if entry > Decimal("0") else Decimal("0")
            )
            opened_at_dt = _parse_iso_datetime(str(position.get("opened_at") or ""))
            position_age_minutes = 0
            if opened_at_dt is not None:
                age_seconds = (
                    datetime.now(timezone.utc) - opened_at_dt
                ).total_seconds()
                position_age_minutes = max(0, int(age_seconds // 60))
            take_profit_hit = pnl_pct >= cfg.live_take_profit_pct
            stop_loss_hit = pnl_pct <= (Decimal("0") - cfg.live_stop_loss_pct)
            timeout_exit_hit = (
                cfg.live_max_position_minutes > 0
                and position_age_minutes >= cfg.live_max_position_minutes
                and pnl_pct <= cfg.live_timeout_exit_pnl_pct
            )
            should_close = (
                action == "sell"
                or take_profit_hit
                or stop_loss_hit
                or timeout_exit_hit
                or bool(news_guard.get("force_exit"))
            )

            if execution_backend == "wallet":
                pinned_network = _normalize_wallet_network(
                    str(position.get("wallet_network", ""))
                )
                if pinned_network:
                    wallet_network = pinned_network

            swap_res = None
            if should_close and execution_backend == "binance_spot":
                token_amount = _as_decimal(position.get("token_amount", "0"))
                sell_qty = (
                    token_amount if token_amount > Decimal("0") else token_free_amount
                )
                if sell_qty > Decimal("0"):
                    order_args = {
                        "symbol": symbol,
                        "side": "SELL",
                        "type": "MARKET",
                        "quantity": str(
                            sell_qty.quantize(
                                Decimal("0.00000001"), rounding=ROUND_DOWN
                            )
                        ),
                        "test": cfg.live_binance_test_order,
                    }
                    if cfg.confirm_code:
                        order_args["confirm_code"] = cfg.confirm_code
                    swap_res = _wallet("binance_order", order_args)
            elif should_close and execution_backend == "metamask_perps":
                close_args: Dict[str, Any] = {
                    "symbol": symbol,
                    "size_pct": "100",
                }
                if cfg.confirm_code:
                    close_args["confirm_code"] = cfg.confirm_code
                swap_res = _wallet("metamask_perps_close_position", close_args)
            elif should_close and execution_backend == "hyperliquid_perps":
                close_args = {
                    "symbol": symbol,
                    "size_pct": "100",
                }
                if cfg.confirm_code:
                    close_args["confirm_code"] = cfg.confirm_code
                swap_res = _wallet("hyperliquid_perps_close_position", close_args)
            elif should_close and asset == "ETH":
                sell_eth = max(Decimal("0"), native_eth - cfg.min_gas_eth)
                if sell_eth > Decimal("0"):
                    swap_args = {
                        "amount_eth": str(
                            sell_eth.quantize(
                                Decimal("0.000000000000000001"), rounding=ROUND_DOWN
                            )
                        ),
                        "slippage_pct": cfg.live_slippage_pct,
                    }
                    if cfg.confirm_code:
                        swap_args["confirm_code"] = cfg.confirm_code
                    swap_res = _wallet("swap_eth_to_usdc", swap_args)
                elif execution_backend == "wallet":
                    # Reconcile stale ETH positions when no sellable ETH remains.
                    # Without this, a timed-out ETH position can stay in position_hold forever.
                    event = "live_close_reconciled"
                    hold_reason = ""
                    live_error = ""
                    position = None
            elif should_close:
                token_amount = _as_decimal(position.get("token_amount", "0"))
                if execution_backend == "wallet":
                    try:
                        bal_args: Dict[str, Any] = {
                            "token_symbol": _wallet_token_symbol(asset)
                        }
                        if wallet_network:
                            bal_args["network"] = wallet_network
                        live_tok = wallet_tool("token_balance", bal_args)
                        if live_tok.get("ok"):
                            token_amount = _as_decimal(live_tok.get("amount", "0"))
                    except Exception:
                        # Keep position token_amount as fallback when balance probe fails.
                        pass
                if token_amount > Decimal("0"):
                    swap_args = {
                        "token_symbol": _wallet_token_symbol(asset),
                        "amount_token": str(token_amount),
                        "slippage_pct": cfg.live_slippage_pct,
                    }
                    if cfg.confirm_code:
                        swap_args["confirm_code"] = cfg.confirm_code
                    swap_res = _wallet("swap_token_to_usdc", swap_args)
                elif execution_backend == "wallet":
                    # Position exists in state, but wallet has no token balance: reconcile
                    # as already closed to avoid duplicate close attempts across near-simultaneous ticks.
                    event = "live_close_reconciled"
                    position = None

            if swap_res is not None:
                if swap_res.get("ok"):
                    if (
                        execution_backend == "binance_spot"
                        and cfg.live_binance_test_order
                    ):
                        order = dict(swap_res.get("order") or {})
                        event = "live_close_test"
                        tx_hash = str(order.get("orderId", ""))
                        learning_note = f"binance_test_order: sell {symbol} qty={position.get('token_amount', '0')}"
                    else:
                        pnl = (usdc_in * pnl_pct).quantize(
                            Decimal("0.000001"), rounding=ROUND_DOWN
                        )
                        close_pnl_pct = pnl_pct
                        realized += pnl
                        event = (
                            "live_close_long_tp"
                            if take_profit_hit
                            else "live_close_long_sl"
                            if stop_loss_hit
                            else "live_close_long"
                            if news_guard.get("force_exit") or action == "sell"
                            else "live_close_long_timeout"
                            if timeout_exit_hit
                            else "live_close_long"
                        )
                        if execution_backend == "binance_spot":
                            order = dict(swap_res.get("order") or {})
                            tx_hash = str(order.get("orderId", ""))
                        elif execution_backend in {
                            "metamask_perps",
                            "hyperliquid_perps",
                        }:
                            closed = dict(swap_res.get("closed") or {})
                            tx_hash = str(
                                closed.get("close_id")
                                or closed.get("order_id")
                                or closed.get("id")
                                or ""
                            )
                        else:
                            tx_hash = str(swap_res.get("swap_hash", ""))
                        trades.append(
                            {
                                "ts": now_iso,
                                "type": "close_long",
                                "symbol": symbol,
                                "price": str(price),
                                "pnl_usdc_est": str(pnl),
                                "tx_hash": tx_hash,
                            }
                        )
                        position = None
                else:
                    event = "live_close_failed"
                    live_error = str(swap_res.get("error", "unknown"))
                    hold_reason = "close_failed"
                    if (
                        execution_backend == "wallet"
                        and "nonce too low" in live_error.lower()
                    ):
                        try:
                            bal_args = {"token_symbol": _wallet_token_symbol(asset)}
                            if wallet_network:
                                bal_args["network"] = wallet_network
                            live_tok = wallet_tool("token_balance", bal_args)
                            live_amt = _as_decimal(
                                (live_tok or {}).get("amount", "0"),
                                "0",
                            )
                            if live_tok.get("ok") and live_amt <= Decimal("0"):
                                event = "live_close_reconciled"
                                live_error = ""
                                hold_reason = ""
                                position = None
                        except Exception:
                            pass

        if event == "live_hold" and not hold_reason:
            if not position and opportunity_score < adaptive_min_opportunity_score:
                hold_reason = "opportunity_below_min"
            elif (
                not position
                and action == "hold"
                and regime == "bear"
                and ai_conf < 0.9
                and cfg.live_score_regime_weight > Decimal("0")
            ):
                hold_reason = "bear_regime_filter"
            elif not position and action == "hold":
                hold_reason = "signal_hold"
            elif position and action == "hold":
                hold_reason = "position_hold"

        hold_streak = int(live_hold_streak_map.get(symbol, 0) or 0)
        if event == "live_hold":
            hold_streak += 1
        else:
            hold_streak = 0

        last_action_ts = int(live_last_action_ts_map.get(symbol, 0) or 0)

        if cfg.live_learning_enabled:
            min_m = max(Decimal("0.0005"), cfg.live_learning_min_momentum_pct)
            max_m = max(min_m, cfg.live_learning_max_momentum_pct)
            step = max(Decimal("0.0001"), cfg.live_learning_step_pct)

            if (
                event == "live_hold"
                and cfg.live_min_actions_per_hour > 0
                and hold_streak > 0
                and hold_streak % cfg.live_learning_hold_streak == 0
            ):
                entry_momentum = max(min_m, entry_momentum - step)
                exit_momentum = max(min_m, exit_momentum - step)
                learning_note = (
                    f"idle_adapt: hold_streak={hold_streak} "
                    f"entry={entry_momentum} exit={exit_momentum}"
                )
            elif event in {
                "live_close_long_sl",
                "live_close_failed",
            } or close_pnl_pct < Decimal("0"):
                entry_momentum = min(max_m, entry_momentum + step)
                exit_momentum = max(min_m, exit_momentum - step)
                learning_note = (
                    f"loss_adapt: entry={entry_momentum} exit={exit_momentum}"
                )
            elif event in {
                "live_close_long_tp",
                "live_close_long",
            } and close_pnl_pct > Decimal("0"):
                entry_momentum = max(min_m, entry_momentum - step)
                exit_momentum = min(max_m, exit_momentum + step)
                learning_note = (
                    f"win_adapt: entry={entry_momentum} exit={exit_momentum}"
                )

            # Activity target: if too much time without actions, relax thresholds a bit.
            if event == "live_hold" and cfg.live_min_actions_per_hour > 0:
                target_gap = max(300, int(3600 / max(1, cfg.live_min_actions_per_hour)))
                since_last_action = (
                    now_ts - last_action_ts if last_action_ts > 0 else target_gap + 1
                )
                if since_last_action > target_gap:
                    boost = max(Decimal("0.0001"), cfg.live_activity_boost_step_pct)
                    entry_momentum = max(min_m, entry_momentum - boost)
                    exit_momentum = max(min_m, exit_momentum - boost)
                    extra = (
                        f"activity_boost: gap={since_last_action}s>{target_gap}s "
                        f"entry={entry_momentum} exit={exit_momentum}"
                    )
                    learning_note = (
                        f"{learning_note}; {extra}" if learning_note else extra
                    )

        if event in {
            "live_close_long_sl",
            "live_close_failed",
        } or close_pnl_pct < Decimal("0"):
            consecutive_losses += 1
        elif event in {
            "live_close_long_tp",
            "live_close_long",
        } and close_pnl_pct > Decimal("0"):
            consecutive_losses = 0
        risk_state["consecutive_losses"] = consecutive_losses
        risk_profiles[risk_scope] = risk_state
        state["live_risk_profiles"] = risk_profiles

        live_hold_streak_map[symbol] = hold_streak
        live_entry_momentum_map[symbol] = str(entry_momentum)
        live_exit_momentum_map[symbol] = str(exit_momentum)
        live_price_history_map[symbol] = price_history
        if (
            event != "live_hold"
            and not event.endswith("_failed")
            and not event.endswith("_blocked")
        ):
            live_last_action_ts_map[symbol] = now_ts

        live_last_prices[symbol] = str(price)
        if position:
            live_positions[symbol] = position
        elif not foreign_position:
            live_positions.pop(symbol, None)
        live_realized_map[symbol] = str(realized)

        # Backwards-compatible mirror for the first configured symbol.
        state["live_last_price"] = str(live_last_prices.get(first_symbol, price))
        state["live_position"] = live_positions.get(first_symbol)
        state["live_realized_pnl_usdc"] = str(
            live_realized_map.get(first_symbol, realized)
        )

        state["live_last_prices"] = live_last_prices
        state["live_positions"] = live_positions
        state["live_realized_pnl_usdc_by_symbol"] = live_realized_map
        state["live_hold_streak_by_symbol"] = live_hold_streak_map
        state["live_entry_momentum_pct_by_symbol"] = live_entry_momentum_map
        state["live_exit_momentum_pct_by_symbol"] = live_exit_momentum_map
        state["live_last_action_ts_by_symbol"] = live_last_action_ts_map
        state["live_price_history_by_symbol"] = live_price_history_map
        state["live_trades"] = trades[-100:]

        return {
            "event": event,
            "symbol": symbol,
            "asset": asset,
            "price": str(price),
            "prev_price": str(prev_price),
            "action": action,
            "ai_signal": ai_signal,
            "news_guard": news_guard,
            "live_realized_pnl_usdc": str(realized),
            "live_position": position,
            "tx_hash": tx_hash,
            "error": live_error,
            "hold_reason": hold_reason,
            "learning_note": learning_note,
            "hold_streak": hold_streak,
            "entry_momentum_pct": str(entry_momentum),
            "exit_momentum_pct": str(exit_momentum),
            "regime": regime,
            "pattern_score": str(pattern_score),
            "pattern_reliability": str(pattern_reliability),
            "effective_pattern_score": str(effective_pattern_score),
            "sma_fast": str(sma_fast),
            "sma_slow": str(sma_slow),
            "opportunity_score": str(opportunity_score),
            "adaptive_min_opportunity_score": str(adaptive_min_opportunity_score),
            "risk_drawdown_pct": str(risk_drawdown_pct),
            "risk_stop_until": int(risk_state.get("stop_until", 0) or 0),
            "risk_consecutive_losses": consecutive_losses,
            "equity_usdc": str(equity_usdc),
            "deployed_usdc": str(deployed_usdc),
            "deployable_baseline_usdc": str(deployable_baseline),
            "growth_scale": str(growth_scale),
            "effective_notional_usdc": str(effective_notional),
            "execution_backend": execution_backend,
            "wallet_network": wallet_network,
            "wallet_total_usdc": str(wallet_total_usdc),
            "wallet_network_balances": wallet_network_balances,
            "wallet_network_errors": wallet_network_errors,
            "wallet_unsupported_networks": wallet_unsupported_networks,
        }

    def _ai_overlay_decision(
        self,
        cfg: "TraderConfig",
        *,
        usdc_amount: Decimal,
        target_keep: Decimal,
        profit: Decimal,
        sweeps_today: int,
    ) -> Dict[str, Any]:
        """Optionally ask the configured LLM for sweep ratio + confidence."""
        base = {
            "used": False,
            "should_send": True,
            "payout_ratio": str(cfg.default_payout_ratio),
            "confidence": 1.0,
            "reason": "ai_disabled",
            "blocked": False,
        }
        if not cfg.use_ai:
            return base

        try:
            from openjarvis.core.types import Message, Role

            messages = [
                Message(
                    role=Role.SYSTEM,
                    content=(
                        "You are a risk filter for crypto profit sweeps. "
                        "Return strict JSON only with keys: should_send (bool), "
                        "payout_ratio (0..1 float), confidence (0..1 float), reason (string)."
                    ),
                ),
                Message(
                    role=Role.USER,
                    content=(
                        "Current snapshot:\n"
                        f"- usdc_balance: {usdc_amount}\n"
                        f"- keep_target: {target_keep}\n"
                        f"- available_profit: {profit}\n"
                        f"- sweeps_today: {sweeps_today}\n"
                        f"- default_payout_ratio: {cfg.default_payout_ratio}\n"
                        "Recommend whether to send now and the payout ratio."
                    ),
                ),
            ]
            result = self._ollama_generate(cfg, messages)
            if result.get("_ollama_error") and not (
                getattr(self, "_engine", None) and getattr(self, "_model", "")
            ):
                base["reason"] = "engine_or_model_unavailable"
                return base
            if not result.get("content") and (
                getattr(self, "_engine", None) and getattr(self, "_model", "")
            ):
                result = self._generate(messages)
            payload = _extract_json_object(str(result.get("content", "")))

            should_send = bool(payload.get("should_send", True))
            conf = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
            ratio = _as_decimal(payload.get("payout_ratio", cfg.default_payout_ratio))
            ratio = max(Decimal("0"), min(cfg.ai_max_payout_ratio, ratio))
            reason = str(payload.get("reason", "ai_ok"))[:280]

            blocked = False
            if conf < cfg.ai_min_confidence:
                blocked = True
                reason = (
                    f"low_confidence:{conf:.2f}<min:{cfg.ai_min_confidence:.2f}; "
                    f"{reason}"
                )

            return {
                "used": True,
                "should_send": should_send,
                "payout_ratio": str(ratio),
                "confidence": conf,
                "reason": reason,
                "blocked": blocked,
            }
        except Exception as exc:
            base["reason"] = f"ai_error:{type(exc).__name__}"
            return base

    def _maybe_autosweep_live(
        self,
        cfg: "TraderConfig",
        state: Dict[str, Any],
        wallet_tool,
    ) -> Dict[str, Any]:
        """Attempt profit sweep in live mode when guardrails allow it.

        Live trading previously returned before reaching the legacy sweep path,
        so this helper mirrors the essential checks for continuous operation.
        """
        result: Dict[str, Any] = {"sent": False, "action": "skip"}

        if not cfg.binance_address:
            result["action"] = "missing_binance_address"
            return result
        if not cfg.autosweep_enabled:
            result["action"] = "autosweep_disabled"
            return result

        configured_networks = _live_wallet_networks(cfg)
        unsupported_networks: list[str] = []

        per_network: list[dict[str, Any]] = []
        for network in configured_networks:
            balance = wallet_tool("balance", {"network": network})
            if not balance.get("ok"):
                err_text = str(balance.get("error", "unknown"))
                if "unsupported network" in err_text.lower():
                    unsupported_networks.append(network)
                continue
            usdc_amount = _as_decimal((balance.get("usdc") or {}).get("amount", "0"))
            native_eth = _as_decimal((balance.get("native") or {}).get("eth", "0"))
            per_network.append(
                {
                    "network": network,
                    "usdc_amount": usdc_amount,
                    "native_eth": native_eth,
                }
            )

        if not per_network:
            result["action"] = "balance_error"
            result["error"] = "no_supported_network_balance"
            result["unsupported_networks"] = unsupported_networks
            return result

        target_keep = cfg.base_capital_usdc + cfg.reserve_usdc
        per_network.sort(key=lambda row: row["usdc_amount"], reverse=True)
        chosen = per_network[0]
        usdc_amount = chosen["usdc_amount"]
        native_eth = chosen["native_eth"]
        sweep_network = str(chosen["network"])
        result["network"] = sweep_network
        result["unsupported_networks"] = unsupported_networks

        target_keep = cfg.base_capital_usdc + cfg.reserve_usdc
        profit = usdc_amount - target_keep
        result["profit_usdc"] = str(profit)

        if profit <= Decimal("0"):
            result["action"] = "no_profit"
            return result
        if profit < cfg.min_sweep_usdc:
            result["action"] = "below_min"
            return result
        if cfg.min_gas_eth > Decimal("0") and native_eth < cfg.min_gas_eth:
            result["action"] = "gas_low"
            result["native_eth"] = str(native_eth)
            return result

        now = int(datetime.now(timezone.utc).timestamp())
        last_sweep = int(state.get("last_sweep_at") or 0)
        if cfg.sweep_cooldown_sec > 0 and now - last_sweep < cfg.sweep_cooldown_sec:
            result["action"] = "cooldown"
            result["wait_sec"] = cfg.sweep_cooldown_sec - (now - last_sweep)
            return result

        day = _today_utc()
        sweeps_today = (
            int(state.get("sweeps_today") or 0) if state.get("day") == day else 0
        )
        if sweeps_today >= cfg.max_sweeps_per_day:
            result["action"] = "daily_limit"
            return result

        ai_decision = self._ai_overlay_decision(
            cfg,
            usdc_amount=usdc_amount,
            target_keep=target_keep,
            profit=profit,
            sweeps_today=sweeps_today,
        )
        result["ai_decision"] = ai_decision

        if ai_decision.get("used") and ai_decision.get("blocked"):
            result["action"] = "ai_blocked"
            return result
        if not bool(ai_decision.get("should_send", True)):
            result["action"] = "ai_hold"
            return result

        payout_ratio = _as_decimal(
            ai_decision.get("payout_ratio", cfg.default_payout_ratio)
        )
        payout_ratio = max(Decimal("0"), min(Decimal("1"), payout_ratio))
        amount_to_send = (profit * payout_ratio).quantize(
            Decimal("0.000001"), rounding=ROUND_DOWN
        )
        if amount_to_send <= Decimal("0"):
            result["action"] = "ratio_zero"
            return result
        if amount_to_send < cfg.min_sweep_usdc:
            result["action"] = "below_min_after_ratio"
            return result

        send_args: Dict[str, Any] = {
            "to": cfg.binance_address,
            "amount_usdc": str(amount_to_send),
            "network": sweep_network,
        }
        if cfg.confirm_code:
            send_args["confirm_code"] = cfg.confirm_code

        sent = wallet_tool("send_usdc", send_args)
        if not sent.get("ok"):
            result["action"] = "send_failed"
            result["error"] = str(sent.get("error", "unknown"))
            return result

        state["last_sweep_at"] = now
        state["last_tx_hash"] = sent.get("tx_hash", "")
        state["day"] = day
        state["sweeps_today"] = sweeps_today + 1
        _save_state(state)

        result.update(
            {
                "sent": True,
                "action": "sent",
                "sent_usdc": str(amount_to_send),
                "network": sweep_network,
                "tx_hash": str(sent.get("tx_hash", "")),
                "explorer_url": str(sent.get("explorer_url", "")),
            }
        )
        return result

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Run one serialized trading tick.

        Holding the lock across wallet execution and state persistence is
        intentional: releasing it earlier could allow a second tick to submit
        an order based on stale balances.
        """
        with _TRADING_TICK_LOCK:
            return self._run_once(input, context=context, **kwargs)

    def _run_once(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        # input/context are accepted for framework compatibility but not required.
        del input, kwargs

        from openjarvis.server.wallet import wallet_tool

        cfg = _load_cfg()
        if context and isinstance(getattr(context, "metadata", None), dict):
            agent_cfg = context.metadata.get("agent_config")
            if isinstance(agent_cfg, dict):
                cfg = _cfg_with_overrides(cfg, agent_cfg)
        state = _load_state()

        if cfg.strategy_mode == "paper":
            try:
                step = self._paper_trade_step(cfg, state)
            except Exception as exc:
                return AgentResult(
                    content=f"Crypto-Trader paper mode error: {exc}",
                    turns=1,
                    metadata={"action": "paper_error", "error": str(exc)[:300]},
                )

            _save_state(state)
            realized = _as_decimal(step.get("paper_realized_pnl_usdc", "0"))
            md = {
                "action": step.get("event", "paper_hold"),
                "strategy_mode": "paper",
                "symbol": cfg.paper_symbol,
                "price": step.get("price", "0"),
                "signal": step.get("action", "hold"),
                "paper_realized_pnl_usdc": str(realized),
                "paper_position": step.get("paper_position"),
                "ai_signal": step.get("ai_signal"),
            }

            # Safe default: do not send real funds from paper profits unless explicitly enabled.
            if not cfg.paper_allow_real_sweep:
                return AgentResult(
                    content=(
                        "Crypto-Trader (paper): simulacion ejecutada. "
                        f"evento={step.get('event')} symbol={cfg.paper_symbol} "
                        f"realized_pnl={realized} USDC."
                    ),
                    turns=1,
                    metadata=md,
                )

        if cfg.strategy_mode == "live":
            steps: list[Dict[str, Any]] = []
            symbols = _live_symbols(cfg)
            live_wallet_networks = _live_wallet_networks(cfg)
            wallet_coverage: Dict[str, Any] = {}
            fallback_backend = "none"
            fallback_error = ""
            wallet_unsupported_assets: set[str] = set()
            configured_fallback = (
                str(cfg.live_wallet_fallback_backend or "none").strip().lower()
            )
            if cfg.live_execution_backend == "wallet" and configured_fallback not in {
                "",
                "none",
                "wallet",
            }:
                try:
                    wallet_coverage = self._wallet_coverage_snapshot(cfg, wallet_tool)
                except Exception as exc:
                    wallet_coverage = {
                        "error": f"coverage_snapshot_error:{type(exc).__name__}"
                    }
                last_good_coverage = state.get("live_wallet_coverage_last_good")
                snapshot_complete = bool(wallet_coverage.get("supported_networks"))
                if snapshot_complete:
                    state["live_wallet_coverage_last_good"] = dict(wallet_coverage)
                elif isinstance(last_good_coverage, dict) and last_good_coverage:
                    wallet_coverage = dict(last_good_coverage)
                    wallet_coverage["stale"] = True
                    wallet_coverage["stale_reason"] = "no_supported_network_snapshot"
                active_wallet_network = _normalize_wallet_network(
                    str(cfg.live_wallet_network or "")
                )
                if not active_wallet_network:
                    active_wallet_network = _normalize_wallet_network(
                        str(state.get("live_wallet_selected_network") or "")
                    )
                if not active_wallet_network and live_wallet_networks:
                    active_wallet_network = _normalize_wallet_network(
                        str(live_wallet_networks[0])
                    )

                network_status = dict(wallet_coverage.get("network_status") or {})
                network_unsupported: list[str] = []
                if active_wallet_network and isinstance(
                    network_status.get(active_wallet_network), dict
                ):
                    network_unsupported = list(
                        (network_status.get(active_wallet_network) or {}).get(
                            "unsupported_requested"
                        )
                        or []
                    )

                raw_unsupported = (
                    network_unsupported
                    if network_unsupported
                    else list(wallet_coverage.get("unsupported_tokens_detected") or [])
                )
                wallet_unsupported_assets = {
                    _wallet_token_symbol(str(s).strip().upper())
                    for s in raw_unsupported
                    if str(s).strip()
                }
                # Never gate core base assets here: they can be tradable via
                # native or dedicated swap paths even when token metadata is noisy.
                wallet_unsupported_assets -= {"BNB", "ETH", "BTC", "MATIC", "POL"}
                fallback_backend = (
                    str(cfg.live_wallet_fallback_backend or "none").strip().lower()
                )
                if fallback_backend == "wallet":
                    fallback_backend = "none"
                if fallback_backend == "binance_spot":
                    acct = wallet_tool("binance_account", {"non_zero_only": True})
                    if not acct.get("ok"):
                        fallback_error = str(acct.get("error", "unknown"))
                        fallback_backend = "none"
                elif fallback_backend == "metamask_perps":
                    acct = wallet_tool("metamask_perps_account", {})
                    if not acct.get("ok"):
                        fallback_error = str(acct.get("error", "unknown"))
                        fallback_backend = "none"
                elif fallback_backend == "hyperliquid_perps":
                    acct = wallet_tool("hyperliquid_perps_account", {})
                    if not acct.get("ok"):
                        fallback_error = str(acct.get("error", "unknown"))
                        fallback_backend = "none"
            elif cfg.live_execution_backend == "metamask_perps":
                perps_account = wallet_tool("metamask_perps_account", {})
                if not perps_account.get("ok"):
                    fallback_error = str(perps_account.get("error", "unknown"))
                    fallback_backend = (
                        str(cfg.live_wallet_fallback_backend or "none").strip().lower()
                    )
                    if fallback_backend in {"none", "wallet", "metamask_perps"}:
                        fallback_backend = "none"
                    if fallback_backend == "binance_spot":
                        acct = wallet_tool("binance_account", {"non_zero_only": True})
                        if not acct.get("ok"):
                            fallback_error = f"{fallback_error}; binance_fallback={acct.get('error', 'unknown')}"
                            fallback_backend = "none"
            elif cfg.live_execution_backend == "hyperliquid_perps":
                perps_account = wallet_tool("hyperliquid_perps_account", {})
                if not perps_account.get("ok"):
                    fallback_error = str(perps_account.get("error", "unknown"))
                    fallback_backend = (
                        str(cfg.live_wallet_fallback_backend or "none").strip().lower()
                    )
                    if fallback_backend in {
                        "none",
                        "wallet",
                        "hyperliquid_perps",
                    }:
                        fallback_backend = "none"
                    if fallback_backend == "binance_spot":
                        acct = wallet_tool("binance_account", {"non_zero_only": True})
                        if not acct.get("ok"):
                            fallback_error = f"{fallback_error}; binance_fallback={acct.get('error', 'unknown')}"
                            fallback_backend = "none"
            news_snapshot = self._fetch_news_feed(cfg)
            news_items = list(news_snapshot.get("items") or [])
            ranked = self._rank_live_symbols(
                cfg,
                state=state,
                symbols=symbols,
                news_items=news_items,
            )
            ranked_by_symbol = {r.get("symbol", ""): r for r in ranked}
            ranked_symbols = [
                str(r.get("symbol", "")) for r in ranked if r.get("symbol")
            ]
            if ranked_symbols:
                ranked_set = {s for s in ranked_symbols}
                missing_symbols = [s for s in symbols if s not in ranked_set]
                symbols = ranked_symbols + missing_symbols
            actions_done = 0
            balance_snapshot: Optional[Dict[str, Any]] = None
            fallback_routes: list[Dict[str, str]] = []

            def _requires_balance_refresh(event: str) -> bool:
                return event in {
                    "live_open_long",
                    "live_close_long",
                    "live_close_long_tp",
                    "live_close_long_sl",
                }

            for symbol in symbols:
                try:
                    execution_backend_override = None
                    wallet_asset = _wallet_token_symbol(_asset_from_symbol(symbol))
                    if (
                        cfg.live_execution_backend == "wallet"
                        and fallback_backend != "none"
                        and wallet_asset in wallet_unsupported_assets
                    ):
                        execution_backend_override = fallback_backend
                        fallback_routes.append(
                            {
                                "symbol": symbol,
                                "asset": _asset_from_symbol(symbol),
                                "backend": fallback_backend,
                            }
                        )
                    elif (
                        cfg.live_execution_backend == "metamask_perps"
                        and fallback_backend != "none"
                    ):
                        execution_backend_override = fallback_backend
                        fallback_routes.append(
                            {
                                "symbol": symbol,
                                "asset": _asset_from_symbol(symbol),
                                "backend": fallback_backend,
                            }
                        )
                    elif (
                        cfg.live_execution_backend == "hyperliquid_perps"
                        and fallback_backend != "none"
                    ):
                        execution_backend_override = fallback_backend
                        fallback_routes.append(
                            {
                                "symbol": symbol,
                                "asset": _asset_from_symbol(symbol),
                                "backend": fallback_backend,
                            }
                        )
                    if balance_snapshot is None and cfg.live_execution_backend in {
                        "wallet",
                        "binance_spot",
                    }:
                        bal_args: Dict[str, Any] = {}
                        if (
                            cfg.live_wallet_network
                            and cfg.live_execution_backend == "wallet"
                            and len(live_wallet_networks) <= 1
                        ):
                            bal_args["network"] = cfg.live_wallet_network
                        balance_snapshot = wallet_tool("balance", bal_args)
                    step = self._live_trade_step(
                        cfg,
                        state,
                        wallet_tool,
                        symbol,
                        news_items,
                        balance_snapshot=balance_snapshot,
                        wallet_network=cfg.live_wallet_network,
                        wallet_networks=live_wallet_networks,
                        pre_news_guard=(ranked_by_symbol.get(symbol) or {}).get(
                            "news_guard"
                        ),
                        rank_score=_as_decimal(
                            (ranked_by_symbol.get(symbol) or {}).get("score", "0")
                        ),
                        execution_backend_override=execution_backend_override,
                    )
                except Exception as exc:
                    logger.exception("Crypto-Trader live tick failed for %s", symbol)
                    error_traceback = traceback.format_exc()
                    return AgentResult(
                        content=f"Crypto-Trader live mode error ({symbol}): {exc}",
                        turns=1,
                        metadata={
                            "action": "live_error",
                            "error": str(exc)[:300],
                            "error_traceback": error_traceback[-4000:],
                            "symbol": symbol,
                        },
                    )
                steps.append(step)
                if _requires_balance_refresh(str(step.get("event", ""))):
                    balance_snapshot = None
                if step.get("event") != "live_hold":
                    actions_done += 1
                    if actions_done >= cfg.live_max_actions_per_tick:
                        break

            _save_state(state)

            sweep_result = self._maybe_autosweep_live(cfg, state, wallet_tool)
            sweep_suffix = ""
            if sweep_result.get("sent"):
                sweep_suffix = (
                    f" | sweep_usdc={sweep_result.get('sent_usdc', '0')}"
                    f" sweep_tx={sweep_result.get('tx_hash', '')}"
                )
            else:
                sweep_action = str(sweep_result.get("action", "skip"))
                sweep_suffix = f" | sweep={sweep_action}"
                if sweep_result.get("error"):
                    sweep_suffix += f" sweep_error={sweep_result.get('error')}"
                if sweep_result.get("wait_sec") is not None:
                    sweep_suffix += f" sweep_wait={sweep_result.get('wait_sec')}s"
                if sweep_result.get("profit_usdc") is not None:
                    sweep_suffix += f" sweep_profit={sweep_result.get('profit_usdc')}"

            friction_cfg = (
                f" | cfg_slip={cfg.live_slippage_pct}"
                f" cfg_fee={cfg.live_micro_tx_fee_ratio}"
                f" cfg_gas_buffer={cfg.live_micro_gas_buffer_usdc}"
                f" cfg_gas_warn={cfg.gas_warning_eth}"
                f" cfg_gas_budget={cfg.gas_refill_usdc_budget}"
            )
            coverage_suffix = ""
            if wallet_coverage:
                if wallet_coverage.get("error"):
                    coverage_suffix = (
                        f" | wallet_cov_error={wallet_coverage.get('error')}"
                    )
                else:
                    if wallet_coverage.get("stale"):
                        coverage_suffix += " | wallet_cov_stale=1"
                    unsupported_networks = list(
                        wallet_coverage.get("unsupported_networks") or []
                    )
                    unsupported_tokens = list(
                        wallet_coverage.get("unsupported_tokens_detected") or []
                    )
                    coverage_suffix = (
                        f" | wallet_cov_tradable={len(wallet_coverage.get('tradable_tokens_detected') or [])}"
                        f" wallet_cov_unsupported={len(wallet_coverage.get('unsupported_tokens_detected') or [])}"
                    )
                    if unsupported_networks:
                        coverage_suffix += " wallet_cov_unsupported_nets=" + ",".join(
                            unsupported_networks[:4]
                        )
                    if unsupported_tokens:
                        coverage_suffix += " wallet_cov_unsupported_tokens=" + ",".join(
                            unsupported_tokens[:6]
                        )
            fallback_suffix = ""
            if fallback_backend != "none":
                fallback_suffix = (
                    f" | fallback_backend={fallback_backend}"
                    f" fallback_routes={len(fallback_routes)}"
                )
            elif fallback_error:
                fallback_suffix = (
                    f" | fallback_backend=none fallback_error={fallback_error}"
                )

            changed = [s for s in steps if s.get("event") != "live_hold"]
            if changed:
                first = changed[0]
                md = {
                    "action": first.get("event", "live_hold"),
                    "strategy_mode": "live",
                    "symbol": first.get("symbol", ""),
                    "asset": first.get("asset", ""),
                    "price": first.get("price", "0"),
                    "signal": first.get("action", "hold"),
                    "live_realized_pnl_usdc": first.get("live_realized_pnl_usdc", "0"),
                    "live_position": first.get("live_position"),
                    "ai_signal": first.get("ai_signal"),
                    "news_guard": first.get("news_guard"),
                    "news_feed_error": news_snapshot.get("error", ""),
                    "tx_hash": first.get("tx_hash", ""),
                    "hold_reason": first.get("hold_reason", ""),
                    "cfg_live_slippage_pct": str(cfg.live_slippage_pct),
                    "cfg_live_micro_tx_fee_ratio": str(cfg.live_micro_tx_fee_ratio),
                    "cfg_live_micro_gas_buffer_usdc": str(
                        cfg.live_micro_gas_buffer_usdc
                    ),
                    "cfg_gas_warning_eth": str(cfg.gas_warning_eth),
                    "cfg_gas_refill_usdc_budget": str(cfg.gas_refill_usdc_budget),
                    "steps": changed,
                    "ranked_symbols": symbols,
                    "sweep": sweep_result,
                    "wallet_coverage": wallet_coverage,
                    "fallback_backend": fallback_backend,
                    "fallback_error": fallback_error,
                    "fallback_routes": fallback_routes,
                }
                if first.get("error"):
                    md["trade_error"] = first.get("error")
                return AgentResult(
                    content=(
                        "Crypto-Trader (live multi): "
                        f"evento={first.get('event')} symbol={first.get('symbol')} "
                        f"price={first.get('price')} tx={first.get('tx_hash', '')}"
                        f" notional={first.get('effective_notional_usdc', cfg.live_notional_usdc)}"
                        f" score={first.get('opportunity_score', '0')}"
                        f" news_score={(first.get('news_guard') or {}).get('score', 0)}"
                        f" risk_dd={first.get('risk_drawdown_pct', '0')}"
                        f" risk_stop_until={first.get('risk_stop_until', 0)}"
                        + (
                            ""
                            if not first.get("hold_reason")
                            else f" hold_reason={first.get('hold_reason')}"
                        )
                        + (
                            ""
                            if not first.get("learning_note")
                            else f" learn={first.get('learning_note')}"
                        )
                        + (
                            ""
                            if not first.get("error")
                            else f" error={first.get('error')}"
                        )
                        + friction_cfg
                        + sweep_suffix
                        + coverage_suffix
                        + fallback_suffix
                        + "."
                    ),
                    turns=1,
                    metadata=md,
                )

            return AgentResult(
                content=(
                    "Crypto-Trader (live multi): sin cambios en este tick. "
                    f"simbolos={','.join(symbols)}"
                    + (
                        ""
                        if not steps
                        else (
                            f" | first={steps[0].get('symbol', '')}"
                            f" action={steps[0].get('action', 'hold')}"
                            f" event={steps[0].get('event', 'live_hold')}"
                            f" price={steps[0].get('price', '0')}"
                            f" prev={steps[0].get('prev_price', '0')}"
                            f" score={steps[0].get('opportunity_score', '0')}"
                            f" news_score={(steps[0].get('news_guard') or {}).get('score', 0)}"
                            f" notional={steps[0].get('effective_notional_usdc', cfg.live_notional_usdc)}"
                            f" risk_dd={steps[0].get('risk_drawdown_pct', '0')}"
                            f" risk_stop_until={steps[0].get('risk_stop_until', 0)}"
                            + (
                                ""
                                if not steps[0].get("hold_reason")
                                else f" hold_reason={steps[0].get('hold_reason')}"
                            )
                            + (
                                ""
                                if not steps[0].get("learning_note")
                                else f" learn={steps[0].get('learning_note')}"
                            )
                            + friction_cfg
                        )
                    )
                    + sweep_suffix
                    + coverage_suffix
                    + fallback_suffix
                ),
                turns=1,
                metadata={
                    "action": "live_hold",
                    "strategy_mode": "live",
                    "symbols": symbols,
                    "ranked_symbols": symbols,
                    "cfg_live_slippage_pct": str(cfg.live_slippage_pct),
                    "cfg_live_micro_tx_fee_ratio": str(cfg.live_micro_tx_fee_ratio),
                    "cfg_live_micro_gas_buffer_usdc": str(
                        cfg.live_micro_gas_buffer_usdc
                    ),
                    "cfg_gas_warning_eth": str(cfg.gas_warning_eth),
                    "cfg_gas_refill_usdc_budget": str(cfg.gas_refill_usdc_budget),
                    "first_hold_reason": (
                        "" if not steps else str(steps[0].get("hold_reason", ""))
                    ),
                    "news_feed_error": news_snapshot.get("error", ""),
                    "steps": steps,
                    "sweep": sweep_result,
                    "wallet_coverage": wallet_coverage,
                    "fallback_backend": fallback_backend,
                    "fallback_error": fallback_error,
                    "fallback_routes": fallback_routes,
                },
            )

        status = wallet_tool("status", {})
        if not status.get("ok"):
            return AgentResult(
                content=f"Crypto-Trader: wallet unavailable: {status.get('error', 'unknown error')}",
                turns=1,
                metadata={"action": "status_error", "status": status},
            )

        if not cfg.binance_address:
            return AgentResult(
                content=(
                    "Crypto-Trader activo en modo monitor, pero falta "
                    "OPENJARVIS_CRYPTO_TRADER_BINANCE_USDC_ADDRESS para enviar beneficios."
                ),
                turns=1,
                metadata={"action": "missing_binance_address"},
            )

        balance = wallet_tool("balance", {})
        if not balance.get("ok"):
            return AgentResult(
                content=f"Crypto-Trader: no se pudo leer balance: {balance.get('error', 'unknown error')}",
                turns=1,
                metadata={"action": "balance_error", "balance": balance},
            )

        usdc_amount = _as_decimal((balance.get("usdc") or {}).get("amount", "0"))
        native_eth = _as_decimal((balance.get("native") or {}).get("eth", "0"))
        target_keep = cfg.base_capital_usdc + cfg.reserve_usdc
        profit = usdc_amount - target_keep

        metadata: Dict[str, Any] = {
            "action": "monitor",
            "usdc_balance": str(usdc_amount),
            "target_keep_usdc": str(target_keep),
            "profit_usdc": str(profit),
            "native_eth": str(native_eth),
            "autosweep_enabled": cfg.autosweep_enabled,
            "default_payout_ratio": str(cfg.default_payout_ratio),
            "min_gas_eth": str(cfg.min_gas_eth),
            "gas_warning_eth": str(cfg.gas_warning_eth),
        }

        if profit <= Decimal("0"):
            return AgentResult(
                content=(
                    f"Crypto-Trader: balance USDC={usdc_amount}. "
                    f"Aun sin beneficio transferible (objetivo a mantener={target_keep})."
                ),
                turns=1,
                metadata=metadata,
            )

        if profit < cfg.min_sweep_usdc:
            return AgentResult(
                content=(
                    f"Crypto-Trader: beneficio detectado {profit} USDC, "
                    f"por debajo del minimo de envio ({cfg.min_sweep_usdc} USDC)."
                ),
                turns=1,
                metadata=metadata,
            )

        if not cfg.autosweep_enabled:
            return AgentResult(
                content=(
                    f"Crypto-Trader: beneficio listo para envio ({profit} USDC) hacia Binance, "
                    "pero autosweep esta desactivado (OPENJARVIS_CRYPTO_TRADER_AUTOSWEEP_ENABLED=0)."
                ),
                turns=1,
                metadata=metadata,
            )

        if cfg.gas_warning_eth > Decimal("0") and native_eth < cfg.gas_warning_eth:
            metadata["gas_warning"] = True

        # --- Auto-recharge: buy ETH with USDC when below warning threshold ---
        if (
            cfg.gas_refill_enabled
            and cfg.gas_warning_eth > Decimal("0")
            and native_eth < cfg.gas_warning_eth
        ):
            # Only spend USDC if we have actual profit above reserve; cap by budget
            available_for_refill = max(Decimal("0"), usdc_amount - target_keep)
            refill_usdc = min(cfg.gas_refill_usdc_budget, available_for_refill)
            if refill_usdc > Decimal("0.50"):  # minimum worth executing
                swap_args: Dict[str, Any] = {
                    "amount_usdc": str(
                        refill_usdc.quantize(Decimal("0.000001"), rounding=ROUND_DOWN)
                    ),
                }
                if cfg.confirm_code:
                    swap_args["confirm_code"] = cfg.confirm_code
                swap_result = wallet_tool("swap_usdc_to_eth", swap_args)
                metadata["gas_refill_attempt"] = swap_result
                if swap_result.get("ok"):
                    metadata["action"] = "gas_refill_sent"
                    return AgentResult(
                        content=(
                            f"Crypto-Trader: gas bajo ({native_eth} ETH < umbral {cfg.gas_warning_eth}). "
                            f"Comprando ~{refill_usdc} USDC de ETH. tx={swap_result.get('swap_hash', '')}. "
                            "Se reanudara el sweep en el proximo tick."
                        ),
                        turns=1,
                        metadata=metadata,
                    )
                else:
                    # Swap failed: still block sweep if below hard minimum
                    metadata["gas_refill_error"] = swap_result.get("error", "unknown")

        if cfg.min_gas_eth > Decimal("0") and native_eth < cfg.min_gas_eth:
            metadata["action"] = "gas_low"
            metadata["gas_warning"] = True
            return AgentResult(
                content=(
                    "Crypto-Trader: envio pausado por gas bajo. "
                    f"ETH actual={native_eth}, minimo requerido={cfg.min_gas_eth}."
                ),
                turns=1,
                metadata=metadata,
            )

        now = int(datetime.now(timezone.utc).timestamp())
        last_sweep = int(state.get("last_sweep_at") or 0)
        if cfg.sweep_cooldown_sec > 0 and now - last_sweep < cfg.sweep_cooldown_sec:
            wait_sec = cfg.sweep_cooldown_sec - (now - last_sweep)
            metadata["action"] = "cooldown"
            metadata["wait_sec"] = wait_sec
            return AgentResult(
                content=(
                    f"Crypto-Trader: beneficio detectado ({profit} USDC), "
                    f"en cooldown ({wait_sec}s restantes) antes del siguiente envio."
                ),
                turns=1,
                metadata=metadata,
            )

        day = _today_utc()
        sweeps_today = (
            int(state.get("sweeps_today") or 0) if state.get("day") == day else 0
        )
        if sweeps_today >= cfg.max_sweeps_per_day:
            metadata["action"] = "daily_limit"
            metadata["sweeps_today"] = sweeps_today
            return AgentResult(
                content=(
                    "Crypto-Trader: beneficio detectado pero limite diario de envios alcanzado "
                    f"({cfg.max_sweeps_per_day})."
                ),
                turns=1,
                metadata=metadata,
            )

        ai_decision = self._ai_overlay_decision(
            cfg,
            usdc_amount=usdc_amount,
            target_keep=target_keep,
            profit=profit,
            sweeps_today=sweeps_today,
        )
        metadata["ai_decision"] = ai_decision

        if ai_decision.get("used") and ai_decision.get("blocked"):
            metadata["action"] = "ai_blocked"
            return AgentResult(
                content=(
                    "Crypto-Trader: envio pausado por filtro IA "
                    f"({ai_decision.get('reason', 'unknown')})."
                ),
                turns=1,
                metadata=metadata,
            )

        should_send = bool(ai_decision.get("should_send", True))
        if not should_send:
            metadata["action"] = "ai_hold"
            return AgentResult(
                content=(
                    "Crypto-Trader: IA recomienda no enviar en este tick "
                    f"({ai_decision.get('reason', 'hold')})."
                ),
                turns=1,
                metadata=metadata,
            )

        payout_ratio = _as_decimal(
            ai_decision.get("payout_ratio", cfg.default_payout_ratio)
        )
        payout_ratio = max(Decimal("0"), min(Decimal("1"), payout_ratio))
        amount_to_send = (profit * payout_ratio).quantize(
            Decimal("0.000001"), rounding=ROUND_DOWN
        )
        metadata["payout_ratio"] = str(payout_ratio)

        if amount_to_send <= Decimal("0"):
            metadata["action"] = "ratio_zero"
            return AgentResult(
                content="Crypto-Trader: ratio de envio en 0, no se realiza transferencia.",
                turns=1,
                metadata=metadata,
            )

        if amount_to_send < cfg.min_sweep_usdc:
            metadata["action"] = "below_min_after_ratio"
            metadata["amount_after_ratio_usdc"] = str(amount_to_send)
            return AgentResult(
                content=(
                    f"Crypto-Trader: monto tras ratio ({amount_to_send} USDC) "
                    f"por debajo del minimo ({cfg.min_sweep_usdc} USDC)."
                ),
                turns=1,
                metadata=metadata,
            )
        send_args: Dict[str, Any] = {
            "to": cfg.binance_address,
            "amount_usdc": str(amount_to_send),
        }
        if cfg.confirm_code:
            send_args["confirm_code"] = cfg.confirm_code

        sent = wallet_tool("send_usdc", send_args)
        if not sent.get("ok"):
            metadata["action"] = "send_failed"
            metadata["send_error"] = sent.get("error", "unknown error")
            return AgentResult(
                content=(
                    f"Crypto-Trader: fallo al enviar beneficio ({amount_to_send} USDC): "
                    f"{sent.get('error', 'unknown error')}"
                ),
                turns=1,
                metadata=metadata,
            )

        state["last_sweep_at"] = now
        state["last_tx_hash"] = sent.get("tx_hash", "")
        state["day"] = day
        state["sweeps_today"] = sweeps_today + 1
        _save_state(state)

        metadata.update(
            {
                "action": "sent",
                "sent_usdc": str(amount_to_send),
                "tx_hash": sent.get("tx_hash", ""),
                "explorer_url": sent.get("explorer_url", ""),
            }
        )
        return AgentResult(
            content=(
                f"Crypto-Trader: enviado beneficio {amount_to_send} USDC a Binance "
                f"({cfg.binance_address}). tx={sent.get('tx_hash', '')}"
            ),
            turns=1,
            metadata=metadata,
        )


__all__ = ["CryptoTraderAgent"]
