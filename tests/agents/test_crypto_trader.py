from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from openjarvis.agents._stubs import AgentContext
from openjarvis.agents.crypto_trader import (
    CryptoTraderAgent,
    _live_risk_scope,
    _pattern_forecast,
    _pattern_reliability,
)


def test_pattern_forecast_detects_persistent_direction():
    rising = [str(100 + index * 0.2) for index in range(60)]
    falling = [str(120 - index * 0.2) for index in range(60)]
    assert _pattern_forecast(rising) > Decimal("0.2")
    assert _pattern_forecast(falling) < Decimal("-0.2")


def test_pattern_forecast_requires_enough_history():
    assert _pattern_forecast(["100", "101", "102"]) == Decimal("0")


def test_pattern_reliability_requires_walk_forward_evidence():
    assert _pattern_reliability([str(100 + index) for index in range(30)]) == 0
    assert _pattern_reliability([str(100 + index) for index in range(120)]) > 0


def test_live_risk_scope_isolates_execution_accounts():
    assert _live_risk_scope("hyperliquid_perps") == "hyperliquid_perps"
    assert _live_risk_scope("metamask_perps") == "metamask_perps"
    assert _live_risk_scope("wallet", "base") == "wallet:base"
    assert _live_risk_scope("wallet", "bnb") == "wallet:bnb"
    assert _live_risk_scope("wallet", "BASE") == "wallet:base"


def _set_default_env(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "OPENJARVIS_CRYPTO_TRADER_STATE_PATH", str(tmp_path / "state.json")
    )
    monkeypatch.setenv(
        "OPENJARVIS_CRYPTO_TRADER_BINANCE_USDC_ADDRESS", "0xBinanceDeposit"
    )
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_BASE_CAPITAL_USDC", "100")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_KEEP_RESERVE_USDC", "10")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_MIN_SWEEP_USDC", "5")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_SWEEP_COOLDOWN_SEC", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_MAX_SWEEPS_PER_DAY", "10")


def test_crypto_trader_missing_binance_address(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "OPENJARVIS_CRYPTO_TRADER_STATE_PATH", str(tmp_path / "state.json")
    )
    monkeypatch.delenv("OPENJARVIS_CRYPTO_TRADER_BINANCE_USDC_ADDRESS", raising=False)

    def _wallet_tool(action, args):
        if action == "status":
            return {"ok": True}
        raise AssertionError("unexpected action")

    monkeypatch.setattr("openjarvis.server.wallet.wallet_tool", _wallet_tool)

    agent = CryptoTraderAgent(engine=None, model="test")
    res = agent.run("tick")

    assert "falta OPENJARVIS_CRYPTO_TRADER_BINANCE_USDC_ADDRESS" in res.content
    assert res.metadata["action"] == "missing_binance_address"


def test_crypto_trader_autosweep_disabled(monkeypatch, tmp_path):
    _set_default_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_AUTOSWEEP_ENABLED", "0")

    def _wallet_tool(action, args):
        if action == "status":
            return {"ok": True}
        if action == "balance":
            return {"ok": True, "native": {"eth": "0.01"}, "usdc": {"amount": "130"}}
        if action == "send_usdc":
            raise AssertionError("send should not be called when autosweep disabled")
        raise AssertionError("unexpected action")

    monkeypatch.setattr("openjarvis.server.wallet.wallet_tool", _wallet_tool)

    agent = CryptoTraderAgent(engine=None, model="test")
    res = agent.run("tick")

    assert "autosweep esta desactivado" in res.content
    assert res.metadata["action"] == "monitor"
    assert Decimal(res.metadata["profit_usdc"]) == Decimal("20")


def test_crypto_trader_send_profit(monkeypatch, tmp_path):
    _set_default_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_AUTOSWEEP_ENABLED", "1")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_CONFIRM_CODE", "go")

    calls = []

    def _wallet_tool(action, args):
        calls.append((action, dict(args)))
        if action == "status":
            return {"ok": True}
        if action == "balance":
            return {"ok": True, "native": {"eth": "0.01"}, "usdc": {"amount": "150.25"}}
        if action == "send_usdc":
            return {
                "ok": True,
                "tx_hash": "0xabc123",
                "explorer_url": "https://basescan.org/tx/0xabc123",
            }
        raise AssertionError("unexpected action")

    monkeypatch.setattr("openjarvis.server.wallet.wallet_tool", _wallet_tool)

    agent = CryptoTraderAgent(engine=None, model="test")
    res = agent.run("tick")

    assert res.metadata["action"] == "sent"
    assert res.metadata["sent_usdc"] == "40.250000"
    assert "tx=0xabc123" in res.content

    send_calls = [c for c in calls if c[0] == "send_usdc"]
    assert len(send_calls) == 1
    assert send_calls[0][1]["to"] == "0xBinanceDeposit"
    assert send_calls[0][1]["amount_usdc"] == "40.250000"
    assert send_calls[0][1]["confirm_code"] == "go"


def test_crypto_trader_ai_enabled_without_engine_falls_back(monkeypatch, tmp_path):
    _set_default_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_AUTOSWEEP_ENABLED", "1")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_USE_AI", "1")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_DEFAULT_PAYOUT_RATIO", "0.5")

    calls = []

    def _wallet_tool(action, args):
        calls.append((action, dict(args)))
        if action == "status":
            return {"ok": True}
        if action == "balance":
            return {"ok": True, "native": {"eth": "0.01"}, "usdc": {"amount": "150.25"}}
        if action == "send_usdc":
            return {"ok": True, "tx_hash": "0xhalf", "explorer_url": "https://x/0xhalf"}
        raise AssertionError("unexpected action")

    monkeypatch.setattr("openjarvis.server.wallet.wallet_tool", _wallet_tool)

    # No engine/model => AI overlay should gracefully fall back.
    agent = CryptoTraderAgent(engine=None, model="")
    res = agent.run("tick")

    assert res.metadata["action"] == "sent"
    assert res.metadata["payout_ratio"] == "0.5"
    assert res.metadata["sent_usdc"] == "20.125000"
    assert res.metadata["ai_decision"]["reason"] == "engine_or_model_unavailable"


def test_crypto_trader_ai_low_confidence_blocks_send(monkeypatch, tmp_path):
    _set_default_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_AUTOSWEEP_ENABLED", "1")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_USE_AI", "1")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_AI_MIN_CONFIDENCE", "0.8")

    class _Engine:
        def generate(self, messages, **kwargs):
            del messages, kwargs
            return {
                "content": (
                    '{"should_send": true, "payout_ratio": 1.0, '
                    '"confidence": 0.2, "reason": "market noisy"}'
                )
            }

    send_called = False

    def _wallet_tool(action, args):
        nonlocal send_called
        if action == "status":
            return {"ok": True}
        if action == "balance":
            return {"ok": True, "native": {"eth": "0.01"}, "usdc": {"amount": "150.25"}}
        if action == "send_usdc":
            send_called = True
            return {"ok": True, "tx_hash": "0xshould_not_happen"}
        raise AssertionError("unexpected action")

    monkeypatch.setattr("openjarvis.server.wallet.wallet_tool", _wallet_tool)

    agent = CryptoTraderAgent(engine=_Engine(), model="test-model")
    res = agent.run("tick")

    assert res.metadata["action"] == "ai_blocked"
    assert "filtro IA" in res.content
    assert send_called is False


def test_crypto_trader_paper_mode_opens_long(monkeypatch, tmp_path):
    _set_default_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_STRATEGY_MODE", "paper")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_PAPER_NOTIONAL_USDC", "100")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_USE_AI", "0")

    prices = [Decimal("100"), Decimal("100.3")]

    def _price(symbol):
        del symbol
        return prices.pop(0)

    monkeypatch.setattr("openjarvis.agents.crypto_trader._fetch_binance_price", _price)

    agent = CryptoTraderAgent(engine=None, model="")
    first = agent.run("tick")
    second = agent.run("tick")

    assert first.metadata["action"] == "paper_hold"
    assert second.metadata["action"] == "paper_open_long"
    assert second.metadata["paper_position"] is not None
    assert "simulacion ejecutada" in second.content


def test_crypto_trader_paper_mode_closes_tp(monkeypatch, tmp_path):
    _set_default_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_STRATEGY_MODE", "paper")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_PAPER_TAKE_PROFIT_PCT", "0.01")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_PAPER_STOP_LOSS_PCT", "0.01")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_USE_AI", "0")

    state_file = tmp_path / "state.json"
    state_file.write_text(
        """
{
  "last_sweep_at": 0,
  "last_tx_hash": "",
  "day": "",
  "sweeps_today": 0,
  "paper_last_price": "100",
  "paper_position": {
    "side": "long",
    "entry_price": "100",
    "qty": "1.0",
    "opened_at": "2026-01-01T00:00:00+00:00"
  },
  "paper_realized_pnl_usdc": "0",
  "paper_trades": []
}
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "openjarvis.agents.crypto_trader._fetch_binance_price",
        lambda _symbol: Decimal("102"),
    )

    agent = CryptoTraderAgent(engine=None, model="")
    res = agent.run("tick")

    assert res.metadata["action"] == "paper_close_long_tp"
    assert Decimal(res.metadata["paper_realized_pnl_usdc"]) == Decimal("2.000000")
    assert res.metadata["paper_position"] is None


def test_crypto_trader_allows_agent_config_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "OPENJARVIS_CRYPTO_TRADER_STATE_PATH", str(tmp_path / "state.json")
    )
    monkeypatch.delenv("OPENJARVIS_CRYPTO_TRADER_BINANCE_USDC_ADDRESS", raising=False)
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_AUTOSWEEP_ENABLED", "0")

    calls = []

    def _wallet_tool(action, args):
        calls.append((action, dict(args)))
        if action == "status":
            return {"ok": True}
        if action == "balance":
            return {"ok": True, "native": {"eth": "0.01"}, "usdc": {"amount": "150"}}
        if action == "send_usdc":
            return {"ok": True, "tx_hash": "0xovr", "explorer_url": "https://x/0xovr"}
        raise AssertionError("unexpected action")

    monkeypatch.setattr("openjarvis.server.wallet.wallet_tool", _wallet_tool)

    ctx = AgentContext()
    ctx.metadata["agent_config"] = {
        "binance_address": "0xCfgBinance",
        "autosweep_enabled": True,
        "base_capital_usdc": "100",
        "reserve_usdc": "10",
        "min_sweep_usdc": "5",
        "sweep_cooldown_sec": 0,
        "max_sweeps_per_day": 10,
    }

    agent = CryptoTraderAgent(engine=None, model="test")
    res = agent.run("tick", context=ctx)

    assert res.metadata["action"] == "sent"
    send_calls = [c for c in calls if c[0] == "send_usdc"]
    assert len(send_calls) == 1
    assert send_calls[0][1]["to"] == "0xCfgBinance"


def test_crypto_trader_successful_send_preserves_existing_state_fields(
    monkeypatch, tmp_path
):
    _set_default_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_AUTOSWEEP_ENABLED", "1")

    state_file = tmp_path / "state.json"
    state_file.write_text(
        """
{
  "last_sweep_at": 0,
  "last_tx_hash": "",
  "day": "",
  "sweeps_today": 0,
  "paper_last_price": "123.45",
  "paper_position": {"side": "long", "entry_price": "120", "qty": "1"},
  "paper_realized_pnl_usdc": "1.5",
  "paper_trades": [{"type": "open_long"}]
}
""".strip(),
        encoding="utf-8",
    )

    def _wallet_tool(action, args):
        if action == "status":
            return {"ok": True}
        if action == "balance":
            return {"ok": True, "native": {"eth": "0.01"}, "usdc": {"amount": "150.25"}}
        if action == "send_usdc":
            return {"ok": True, "tx_hash": "0xkeep", "explorer_url": "https://x/0xkeep"}
        raise AssertionError("unexpected action")

    monkeypatch.setattr("openjarvis.server.wallet.wallet_tool", _wallet_tool)

    agent = CryptoTraderAgent(engine=None, model="test")
    res = agent.run("tick")
    assert res.metadata["action"] == "sent"

    saved = state_file.read_text(encoding="utf-8")
    assert '"paper_last_price": "123.45"' in saved
    assert '"paper_realized_pnl_usdc": "1.5"' in saved


def test_crypto_trader_pauses_send_when_gas_below_min(monkeypatch, tmp_path):
    _set_default_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_AUTOSWEEP_ENABLED", "1")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_MIN_GAS_ETH", "0.01")

    send_called = False

    def _wallet_tool(action, args):
        nonlocal send_called
        if action == "status":
            return {"ok": True}
        if action == "balance":
            return {
                "ok": True,
                "native": {"eth": "0.0005"},
                "usdc": {"amount": "150.25"},
            }
        if action == "send_usdc":
            send_called = True
            return {"ok": True, "tx_hash": "0xshould_not_send"}
        if action == "swap_usdc_to_eth":
            # refill attempt fails: no real DEX in tests
            return {"ok": False, "error": "test_mock_no_dex"}
        raise AssertionError("unexpected action")

    monkeypatch.setattr("openjarvis.server.wallet.wallet_tool", _wallet_tool)

    agent = CryptoTraderAgent(engine=None, model="test")
    res = agent.run("tick")

    assert res.metadata["action"] == "gas_low"
    assert "gas bajo" in res.content
    assert send_called is False


def test_crypto_trader_live_mode_opens_long(monkeypatch, tmp_path):
    _set_default_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_STRATEGY_MODE", "live")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_AUTOSWEEP_ENABLED", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_NOTIONAL_USDC", "20")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MIN_TRADE_USDC", "5")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_USE_AI", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MIN_OPPORTUNITY_SCORE", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MICRO_MIN_NET_EDGE_PCT", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MICRO_TX_FEE_RATIO", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MICRO_GAS_BUFFER_USDC", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_SLIPPAGE_PCT", "0")

    prices = [Decimal("100"), Decimal("100"), Decimal("100.1"), Decimal("100.4")]
    last_price = prices[-1]

    def _price(symbol):
        nonlocal last_price
        del symbol
        if prices:
            last_price = prices.pop(0)
        return last_price

    calls = []

    def _wallet_tool(action, args):
        calls.append((action, dict(args)))
        if action == "status":
            return {"ok": True}
        if action == "balance":
            return {"ok": True, "native": {"eth": "0.01"}, "usdc": {"amount": "110"}}
        if action == "swap_usdc_to_eth":
            return {"ok": True, "swap_hash": "0xliveopen"}
        raise AssertionError("unexpected action")

    monkeypatch.setattr("openjarvis.agents.crypto_trader._fetch_binance_price", _price)
    monkeypatch.setattr("openjarvis.server.wallet.wallet_tool", _wallet_tool)

    agent = CryptoTraderAgent(engine=None, model="")
    first = agent.run("tick")
    second = agent.run("tick")

    assert first.metadata["action"] in {"monitor", "live_hold"}
    assert second.metadata["action"] == "live_open_long"
    assert second.metadata["tx_hash"] == "0xliveopen"
    assert any(action == "swap_usdc_to_eth" for action, _args in calls)


def test_crypto_trader_live_mode_closes_tp(monkeypatch, tmp_path):
    _set_default_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_STRATEGY_MODE", "live")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_AUTOSWEEP_ENABLED", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_TAKE_PROFIT_PCT", "0.01")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_STOP_LOSS_PCT", "0.01")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_USE_AI", "0")

    state_file = tmp_path / "state.json"
    state_file.write_text(
        """
{
  "last_sweep_at": 0,
  "last_tx_hash": "",
  "day": "",
  "sweeps_today": 0,
  "paper_last_price": "0",
  "paper_position": null,
  "paper_realized_pnl_usdc": "0",
  "paper_trades": [],
  "live_last_price": "100",
  "live_position": {
    "side": "long",
    "entry_price": "100",
    "usdc_in": "20",
    "opened_at": "2026-01-01T00:00:00+00:00"
  },
  "live_realized_pnl_usdc": "0",
  "live_trades": []
}
""".strip(),
        encoding="utf-8",
    )

    calls = []

    def _wallet_tool(action, args):
        calls.append((action, dict(args)))
        if action == "status":
            return {"ok": True}
        if action == "balance":
            return {"ok": True, "native": {"eth": "0.02"}, "usdc": {"amount": "90"}}
        if action == "swap_eth_to_usdc":
            return {"ok": True, "swap_hash": "0xliveclose"}
        raise AssertionError("unexpected action")

    monkeypatch.setattr(
        "openjarvis.agents.crypto_trader._fetch_binance_price",
        lambda _symbol: Decimal("102"),
    )
    monkeypatch.setattr("openjarvis.server.wallet.wallet_tool", _wallet_tool)

    agent = CryptoTraderAgent(engine=None, model="")
    res = agent.run("tick")

    assert res.metadata["action"] == "live_close_long_tp"
    assert res.metadata["tx_hash"] == "0xliveclose"
    assert any(action == "swap_eth_to_usdc" for action, _args in calls)


def test_crypto_trader_live_mode_multi_symbols_opens_non_eth(monkeypatch, tmp_path):
    _set_default_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_STRATEGY_MODE", "live")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_SYMBOLS", "BTCUSDT,ETHUSDT")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MAX_ACTIONS_PER_TICK", "1")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_AUTOSWEEP_ENABLED", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_NOTIONAL_USDC", "20")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MIN_TRADE_USDC", "5")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_USE_AI", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MIN_OPPORTUNITY_SCORE", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MICRO_MIN_NET_EDGE_PCT", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MICRO_TX_FEE_RATIO", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MICRO_GAS_BUFFER_USDC", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_SLIPPAGE_PCT", "0")

    prices = [
        Decimal("50000"),
        Decimal("100"),
        Decimal("50000"),
        Decimal("50060"),
        Decimal("50060"),
        Decimal("50090"),
    ]
    last_price = prices[-1]

    def _price(symbol):
        nonlocal last_price
        del symbol
        if prices:
            last_price = prices.pop(0)
        return last_price

    calls = []

    def _wallet_tool(action, args):
        calls.append((action, dict(args)))
        if action == "balance":
            return {"ok": True, "native": {"eth": "0.02"}, "usdc": {"amount": "120"}}
        if action == "swap_usdc_to_token":
            return {"ok": True, "swap_hash": "0xbtcopen", "token_received": "0.0004"}
        raise AssertionError("unexpected action")

    monkeypatch.setattr("openjarvis.agents.crypto_trader._fetch_binance_price", _price)
    monkeypatch.setattr("openjarvis.server.wallet.wallet_tool", _wallet_tool)

    agent = CryptoTraderAgent(engine=None, model="")
    first = agent.run("tick")
    second = agent.run("tick")

    assert first.metadata["action"] == "live_hold"
    assert second.metadata["action"] == "live_open_long"
    assert second.metadata["symbol"] == "BTCUSDT"
    assert second.metadata["asset"] == "BTC"
    assert any(
        a == "swap_usdc_to_token" and b.get("token_symbol") == "BTC" for a, b in calls
    )


def test_crypto_trader_live_mode_news_blocks_buy(monkeypatch, tmp_path):
    _set_default_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_STRATEGY_MODE", "live")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_SYMBOL", "ETHUSDT")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_AUTOSWEEP_ENABLED", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_USE_AI", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_NEWS_BLOCK_BUY_SCORE", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MIN_OPPORTUNITY_SCORE", "0")

    prices = [Decimal("100"), Decimal("100"), Decimal("100.1"), Decimal("100.4")]
    last_price = prices[-1]

    def _price(_symbol):
        nonlocal last_price
        if prices:
            last_price = prices.pop(0)
        return last_price

    calls = []

    def _wallet_tool(action, args):
        calls.append((action, dict(args)))
        if action == "balance":
            return {"ok": True, "native": {"eth": "0.03"}, "usdc": {"amount": "120"}}
        if action == "swap_usdc_to_eth":
            return {"ok": True, "swap_hash": "0xshould_not_happen"}
        raise AssertionError("unexpected action")

    def _news(_self, _cfg):
        return {
            "items": [
                {
                    "title": "Ethereum hacked after major exchange exploit",
                    "summary": "market crash and liquidations follow",
                    "published_at": datetime.now(timezone.utc).isoformat(),
                }
            ]
        }

    monkeypatch.setattr("openjarvis.agents.crypto_trader._fetch_binance_price", _price)
    monkeypatch.setattr("openjarvis.server.wallet.wallet_tool", _wallet_tool)
    monkeypatch.setattr(
        "openjarvis.agents.crypto_trader.CryptoTraderAgent._fetch_news_feed", _news
    )

    agent = CryptoTraderAgent(engine=None, model="")
    first = agent.run("tick")
    second = agent.run("tick")

    assert first.metadata["action"] == "live_hold"
    assert second.metadata["action"] == "live_news_blocked"
    assert not any(action == "swap_usdc_to_eth" for action, _args in calls)


def test_crypto_trader_live_mode_reports_hold_reason_when_buy_below_min_trade(
    monkeypatch, tmp_path
):
    _set_default_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_STRATEGY_MODE", "live")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_SYMBOL", "ETHUSDT")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_AUTOSWEEP_ENABLED", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_USE_AI", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MIN_OPPORTUNITY_SCORE", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MICRO_MIN_NET_EDGE_PCT", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MICRO_TX_FEE_RATIO", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MICRO_GAS_BUFFER_USDC", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MIN_TRADE_USDC", "25")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_NOTIONAL_USDC", "24")

    prices = [Decimal("100"), Decimal("100"), Decimal("100.3")]
    last_price = prices[-1]

    def _price(_symbol):
        nonlocal last_price
        if prices:
            last_price = prices.pop(0)
        return last_price

    def _wallet_tool(action, args):
        del args
        if action == "balance":
            return {"ok": True, "native": {"eth": "0.02"}, "usdc": {"amount": "24"}}
        raise AssertionError("unexpected action")

    monkeypatch.setattr("openjarvis.agents.crypto_trader._fetch_binance_price", _price)
    monkeypatch.setattr("openjarvis.server.wallet.wallet_tool", _wallet_tool)

    agent = CryptoTraderAgent(engine=None, model="")
    first = agent.run("tick")
    second = agent.run("tick")

    assert first.metadata["action"] == "live_hold"
    assert second.metadata["action"] == "live_hold"
    assert second.metadata["first_hold_reason"] == "buy_usdc_below_min_trade"
    assert second.metadata["steps"][0]["action"] == "buy"
    assert second.metadata["steps"][0]["event"] == "live_hold"
    assert second.metadata["steps"][0]["hold_reason"] == "buy_usdc_below_min_trade"


def test_crypto_trader_live_mode_news_forces_exit(monkeypatch, tmp_path):
    _set_default_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_STRATEGY_MODE", "live")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_SYMBOL", "ETHUSDT")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_AUTOSWEEP_ENABLED", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_USE_AI", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_NEWS_FORCE_EXIT_SCORE", "-2")

    state_file = tmp_path / "state.json"
    state_file.write_text(
        """
{
  "last_sweep_at": 0,
  "last_tx_hash": "",
  "day": "",
  "sweeps_today": 0,
  "paper_last_price": "0",
  "paper_position": null,
  "paper_realized_pnl_usdc": "0",
  "paper_trades": [],
  "live_last_price": "100",
  "live_position": {
    "side": "long",
    "entry_price": "100",
    "usdc_in": "20",
    "token_symbol": "ETH",
    "opened_at": "2026-01-01T00:00:00+00:00"
  },
  "live_realized_pnl_usdc": "0",
  "live_trades": []
}
""".strip(),
        encoding="utf-8",
    )

    calls = []

    def _wallet_tool(action, args):
        calls.append((action, dict(args)))
        if action == "balance":
            return {"ok": True, "native": {"eth": "0.03"}, "usdc": {"amount": "90"}}
        if action == "swap_eth_to_usdc":
            return {"ok": True, "swap_hash": "0xnewsclose"}
        raise AssertionError("unexpected action")

    def _news(_self, _cfg):
        return {
            "items": [
                {
                    "title": "SEC lawsuit escalates as crypto market crash deepens",
                    "summary": "liquidation cascade and exchange outage",
                    "published_at": datetime.now(timezone.utc).isoformat(),
                }
            ]
        }

    monkeypatch.setattr(
        "openjarvis.agents.crypto_trader._fetch_binance_price",
        lambda _symbol: Decimal("100.1"),
    )
    monkeypatch.setattr("openjarvis.server.wallet.wallet_tool", _wallet_tool)
    monkeypatch.setattr(
        "openjarvis.agents.crypto_trader.CryptoTraderAgent._fetch_news_feed", _news
    )

    agent = CryptoTraderAgent(engine=None, model="")
    res = agent.run("tick")

    assert res.metadata["action"] == "live_close_long"
    assert res.metadata["tx_hash"] == "0xnewsclose"
    assert any(action == "swap_eth_to_usdc" for action, _args in calls)


def test_crypto_trader_live_mode_forwards_wallet_network(monkeypatch, tmp_path):
    _set_default_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_STRATEGY_MODE", "live")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_SYMBOL", "ETHUSDT")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_WALLET_NETWORK", "bnb")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_AUTOSWEEP_ENABLED", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_USE_AI", "0")

    monkeypatch.setattr(
        "openjarvis.agents.crypto_trader._fetch_binance_price",
        lambda _symbol: Decimal("100"),
    )

    calls = []

    def _wallet_tool(action, args):
        calls.append((action, dict(args)))
        if action == "balance":
            return {"ok": True, "native": {"eth": "0.03"}, "usdc": {"amount": "120"}}
        raise AssertionError("unexpected action")

    monkeypatch.setattr("openjarvis.server.wallet.wallet_tool", _wallet_tool)

    agent = CryptoTraderAgent(engine=None, model="")
    res = agent.run("tick")

    assert res.metadata["action"] == "live_hold"
    assert calls
    assert calls[0][0] == "balance"
    assert calls[0][1].get("network") == "bnb"


def test_crypto_trader_live_mode_metamask_perps_open_and_close(monkeypatch, tmp_path):
    _set_default_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_STRATEGY_MODE", "live")
    monkeypatch.setenv(
        "OPENJARVIS_CRYPTO_TRADER_LIVE_EXECUTION_BACKEND", "metamask_perps"
    )
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_AUTOSWEEP_ENABLED", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_NOTIONAL_USDC", "20")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MIN_TRADE_USDC", "5")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_TAKE_PROFIT_PCT", "0.01")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_STOP_LOSS_PCT", "0.02")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_USE_AI", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MIN_OPPORTUNITY_SCORE", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MICRO_MIN_NET_EDGE_PCT", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MICRO_TX_FEE_RATIO", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MICRO_GAS_BUFFER_USDC", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_SLIPPAGE_PCT", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_PERPS_LEVERAGE", "4")

    prices = [Decimal("100"), Decimal("100"), Decimal("100.3"), Decimal("102")]
    last_price = prices[-1]

    def _price(_symbol):
        nonlocal last_price
        if prices:
            last_price = prices.pop(0)
        return last_price

    calls = []

    def _wallet_tool(action, args):
        calls.append((action, dict(args)))
        if action == "status":
            return {"ok": True}
        if action == "balance":
            return {"ok": True, "native": {"eth": "0.02"}, "usdc": {"amount": "120"}}
        if action == "metamask_perps_account":
            return {"ok": True, "account": {"available_usdc": "120"}}
        if action == "metamask_perps_order":
            return {"ok": True, "order": {"order_id": "perp-open-1"}}
        if action == "metamask_perps_close_position":
            return {"ok": True, "closed": {"close_id": "perp-close-1"}}
        raise AssertionError("unexpected action")

    monkeypatch.setattr("openjarvis.agents.crypto_trader._fetch_binance_price", _price)
    monkeypatch.setattr("openjarvis.server.wallet.wallet_tool", _wallet_tool)

    agent = CryptoTraderAgent(engine=None, model="")
    first = agent.run("tick")
    second = agent.run("tick")
    assert first.metadata["action"] in {"monitor", "live_hold"}
    assert second.metadata["action"] == "live_open_long"
    assert second.metadata["tx_hash"] == "perp-open-1"

    # Force a deterministic take-profit close on the next tick.
    monkeypatch.setattr(
        "openjarvis.agents.crypto_trader._fetch_binance_price",
        lambda _symbol: Decimal("104"),
    )
    third = agent.run("tick")

    assert third.metadata["action"] == "live_close_long_tp"
    assert third.metadata["tx_hash"] == "perp-close-1"
    perps_open_calls = [
        call_args for action, call_args in calls if action == "metamask_perps_order"
    ]
    assert perps_open_calls
    assert perps_open_calls[0]["leverage"] == 4
    assert any(action == "metamask_perps_order" for action, _args in calls)
    assert any(action == "metamask_perps_close_position" for action, _args in calls)


def test_crypto_trader_live_mode_perps_uses_leverage_for_min_order_notional(
    monkeypatch, tmp_path
):
    _set_default_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_STRATEGY_MODE", "live")
    monkeypatch.setenv(
        "OPENJARVIS_CRYPTO_TRADER_LIVE_EXECUTION_BACKEND", "hyperliquid_perps"
    )
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_AUTOSWEEP_ENABLED", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_NOTIONAL_USDC", "8")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MIN_TRADE_USDC", "10")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_PERPS_LEVERAGE", "5")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_USE_AI", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MIN_OPPORTUNITY_SCORE", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MICRO_MIN_NET_EDGE_PCT", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MICRO_TX_FEE_RATIO", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MICRO_GAS_BUFFER_USDC", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_SLIPPAGE_PCT", "0")

    prices = [Decimal("100"), Decimal("100"), Decimal("100.3")]
    last_price = prices[-1]

    def _price(_symbol):
        nonlocal last_price
        if prices:
            last_price = prices.pop(0)
        return last_price

    calls = []

    def _wallet_tool(action, args):
        calls.append((action, dict(args)))
        if action == "status":
            return {"ok": True}
        if action == "hyperliquid_perps_account":
            return {"ok": True, "account": {"available_usdc": "9.268316"}}
        if action == "hyperliquid_perps_order":
            return {"ok": True, "order": {"order_id": "hl-open-1"}}
        raise AssertionError(f"unexpected action {action}")

    monkeypatch.setattr("openjarvis.agents.crypto_trader._fetch_binance_price", _price)
    monkeypatch.setattr("openjarvis.server.wallet.wallet_tool", _wallet_tool)

    agent = CryptoTraderAgent(engine=None, model="")
    first = agent.run("tick")
    second = agent.run("tick")

    assert first.metadata["action"] in {"monitor", "live_hold"}
    assert second.metadata["action"] == "live_open_long"
    perps_open_calls = [
        call_args for action, call_args in calls if action == "hyperliquid_perps_order"
    ]
    assert perps_open_calls
    assert Decimal(perps_open_calls[0]["notional_usdc"]) == Decimal("10.000000")
    assert perps_open_calls[0]["leverage"] == 5


def test_crypto_trader_live_growth_counts_deployed_capital(monkeypatch, tmp_path):
    _set_default_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_STRATEGY_MODE", "live")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_SYMBOL", "ETHUSDT")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_NOTIONAL_USDC", "20")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_GROWTH_ENABLED", "1")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_GROWTH_REINVEST_RATIO", "1")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_GROWTH_MAX_NOTIONAL_USDC", "250")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_USE_AI", "0")

    state_file = tmp_path / "state.json"
    state_file.write_text(
        """
{
  "last_sweep_at": 0,
  "last_tx_hash": "",
  "day": "",
  "sweeps_today": 0,
  "paper_last_price": "0",
  "paper_position": null,
  "paper_realized_pnl_usdc": "0",
  "paper_trades": [],
  "live_last_price": "100",
  "live_position": {
    "side": "long",
    "entry_price": "100",
    "usdc_in": "40",
    "token_symbol": "ETH",
    "opened_at": "2026-01-01T00:00:00+00:00"
  },
  "live_realized_pnl_usdc": "0",
  "live_trades": []
}
""".strip(),
        encoding="utf-8",
    )

    def _wallet_tool(action, args):
        del args
        if action == "balance":
            return {"ok": True, "native": {"eth": "0.03"}, "usdc": {"amount": "70"}}
        if action == "swap_eth_to_usdc":
            return {"ok": True, "swap_hash": "0xclose"}
        raise AssertionError("unexpected action")

    monkeypatch.setattr(
        "openjarvis.agents.crypto_trader._fetch_binance_price",
        lambda _symbol: Decimal("100.05"),
    )
    monkeypatch.setattr("openjarvis.server.wallet.wallet_tool", _wallet_tool)

    agent = CryptoTraderAgent(engine=None, model="")
    res = agent.run("tick")

    # Baseline deployable is 90 (base 100 - reserve 10). Total equity is 70 + 40 - 10 = 100,
    # so growth scale is 1 + (10/90) and notional should be > 22 USDC.
    assert res.metadata["action"] == "live_hold"
    step = res.metadata["steps"][0]
    assert Decimal(step["deployed_usdc"]) == Decimal("40")
    assert Decimal(step["effective_notional_usdc"]) > Decimal("22")


def test_crypto_trader_live_multi_reuses_balance_snapshot_per_tick(
    monkeypatch, tmp_path
):
    _set_default_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_STRATEGY_MODE", "live")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_SYMBOLS", "ETHUSDT,BTCUSDT")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_LIVE_MAX_ACTIONS_PER_TICK", "2")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_AUTOSWEEP_ENABLED", "0")
    monkeypatch.setenv("OPENJARVIS_CRYPTO_TRADER_USE_AI", "0")

    prices = [Decimal("100"), Decimal("50000")]
    last_price = prices[-1]

    def _price(_symbol):
        nonlocal last_price
        if prices:
            last_price = prices.pop(0)
        return last_price

    balance_calls = 0

    def _wallet_tool(action, args):
        del args
        nonlocal balance_calls
        if action == "balance":
            balance_calls += 1
            return {"ok": True, "native": {"eth": "0.03"}, "usdc": {"amount": "120"}}
        raise AssertionError("unexpected action")

    monkeypatch.setattr("openjarvis.agents.crypto_trader._fetch_binance_price", _price)
    monkeypatch.setattr("openjarvis.server.wallet.wallet_tool", _wallet_tool)

    agent = CryptoTraderAgent(engine=None, model="")
    res = agent.run("tick")

    assert res.metadata["action"] == "live_hold"
    assert len(res.metadata["steps"]) == 2
    assert balance_calls == 1
