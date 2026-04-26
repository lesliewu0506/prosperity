from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    """
    Round 3 strategy draft.

    Products:
      - HYDROGEL_PACK: delta-1 mean-reversion / market-making
      - VELVETFRUIT_EXTRACT: delta-1 mean-reversion / market-making + option hedge instrument
      - VEV_* vouchers: call options on VELVETFRUIT_EXTRACT

    Main idea:
      1. EMA fair value for HYD and VEX with inventory-skewed aggressive trading.
      2. Black-Scholes-style theoretical value for central VEV strikes using a fixed fair IV.
      3. Trade vouchers when best bid/ask deviates enough from theoretical value.
      4. Option positions are approximately delta-hedged with VEX, with a per-tick hedge cap.
    """

    DELTA_PARAMS = {
        "HYDROGEL_PACK": {
            "position_limit": 200,
            "default_fair": 10000.0,
            "ema_alpha": 0.10,
            "base_edge": 3.0,
            "vol_edge_mult": 1.2,
            "min_edge": 2.0,
            "max_edge": 8.0,
            "max_skew": 5.0,
            "max_trade_size": 60,
        },
        "VELVETFRUIT_EXTRACT": {
            "position_limit": 200,
            "default_fair": 5250.0,
            "ema_alpha": 0.12,
            "base_edge": 3.0,
            "vol_edge_mult": 1.5,
            "min_edge": 2.0,
            "max_edge": 8.0,
            "max_skew": 5.0,
            "max_trade_size": 60,
        },
    }

    VOUCHER_STRIKES = {
        "VEV_4000": 4000,
        "VEV_4500": 4500,
        "VEV_5000": 5000,
        "VEV_5100": 5100,
        "VEV_5200": 5200,
        "VEV_5300": 5300,
        "VEV_5400": 5400,
        "VEV_5500": 5500,
        "VEV_6000": 6000,
        "VEV_6500": 6500,
    }

    # The central strikes had the cleanest historical IV behaviour.
    ACTIVE_VOUCHERS = {
        "VEV_5000",
        "VEV_5100",
        "VEV_5200",
        "VEV_5300",
        "VEV_5400",
        "VEV_5500",
    }

    PARAMS = {
        "voucher_position_limit": 300,
        "voucher_soft_limit": 180,
        "voucher_max_trade_size": 50,
        "fair_iv": 0.23,
        "tte_days": 5.0,
        "option_min_edge": 1.5,
        "option_edge_fraction": 0.020,
        "option_inventory_skew": 4.0,
        "vex_hedge_fraction": 0.75,
        "vex_max_hedge_trade": 80,
        "return_window": 60,
        "mid_history_cap": 120,
    }

    # ---------- generic utilities ----------
    @staticmethod
    def best_bid_ask(order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    @staticmethod
    def mid_price(order_depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask = Trader.best_bid_ask(order_depth)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return None

    @staticmethod
    def normal_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @classmethod
    def bs_call(cls, S: float, K: float, T_days: float, sigma: float) -> float:
        T = max(T_days, 1e-9) / 365.0
        if S <= 0 or K <= 0 or sigma <= 0:
            return max(S - K, 0.0)
        vol_sqrt_T = sigma * math.sqrt(T)
        if vol_sqrt_T <= 1e-12:
            return max(S - K, 0.0)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / vol_sqrt_T
        d2 = d1 - vol_sqrt_T
        return S * cls.normal_cdf(d1) - K * cls.normal_cdf(d2)

    @classmethod
    def bs_delta(cls, S: float, K: float, T_days: float, sigma: float) -> float:
        T = max(T_days, 1e-9) / 365.0
        if S <= 0 or K <= 0 or sigma <= 0:
            return 1.0 if S > K else 0.0
        vol_sqrt_T = sigma * math.sqrt(T)
        if vol_sqrt_T <= 1e-12:
            return 1.0 if S > K else 0.0
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / vol_sqrt_T
        return cls.normal_cdf(d1)

    @staticmethod
    def rolling_abs_return_vol(mid_history: List[float]) -> float:
        if len(mid_history) < 3:
            return 0.0
        rets = []
        for a, b in zip(mid_history[:-1], mid_history[1:]):
            if a > 0:
                rets.append(abs((b - a) / a))
        if not rets:
            return 0.0
        # Convert percentage-return scale back to price-ish edge scale later by multiplying by current mid.
        return sum(rets[-60:]) / min(len(rets), 60)

    # ---------- delta-1 strategy ----------
    def trade_delta_product(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        saved: Dict,
        new_saved: Dict,
    ) -> Tuple[List[Order], int]:
        params = self.DELTA_PARAMS[product]
        orders: List[Order] = []

        mid = self.mid_price(order_depth)
        if mid is None:
            return orders, position

        fair_values = saved.get("fair_values", {})
        prev_fair = float(fair_values.get(product, params["default_fair"]))
        alpha = params["ema_alpha"]
        fair = (1.0 - alpha) * prev_fair + alpha * mid
        new_saved.setdefault("fair_values", {})[product] = fair

        histories = saved.get("mid_histories", {})
        hist = list(histories.get(product, []))[-self.PARAMS["mid_history_cap"]:]
        hist.append(mid)
        new_saved.setdefault("mid_histories", {})[product] = hist[-self.PARAMS["mid_history_cap"]:]

        rel_vol = self.rolling_abs_return_vol(hist)
        vol_edge = params["vol_edge_mult"] * rel_vol * mid
        edge = min(params["max_edge"], max(params["min_edge"], params["base_edge"] + vol_edge))

        limit = params["position_limit"]
        max_trade = params["max_trade_size"]
        # Nonlinear skew: stronger as position approaches the limit.
        pos_frac = position / limit
        skew = params["max_skew"] * pos_frac * abs(pos_frac)

        buy_threshold = fair - edge - skew
        sell_threshold = fair + edge - skew

        best_bid, best_ask = self.best_bid_ask(order_depth)

        if best_ask is not None and best_ask <= buy_threshold:
            ask_volume = -order_depth.sell_orders[best_ask]
            qty = min(limit - position, ask_volume, max_trade)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                position += qty

        if best_bid is not None and best_bid >= sell_threshold:
            bid_volume = order_depth.buy_orders[best_bid]
            qty = min(limit + position, bid_volume, max_trade)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                position -= qty

        return orders, position

    # ---------- option strategy ----------
    def trade_voucher(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        vex_mid: float,
    ) -> Tuple[List[Order], int]:
        orders: List[Order] = []
        if product not in self.ACTIVE_VOUCHERS:
            return orders, position

        K = self.VOUCHER_STRIKES[product]
        fair_iv = self.PARAMS["fair_iv"]
        tte_days = self.PARAMS["tte_days"]
        theo = self.bs_call(vex_mid, K, tte_days, fair_iv)

        # Avoid trading options whose theoretical value is effectively just the tick floor.
        if theo < 2.0:
            return orders, position

        edge = max(self.PARAMS["option_min_edge"], self.PARAMS["option_edge_fraction"] * theo)
        limit = self.PARAMS["voucher_position_limit"]
        soft_limit = self.PARAMS["voucher_soft_limit"]
        max_trade = self.PARAMS["voucher_max_trade_size"]

        # Position skew: if already long, require a cheaper ask and accept lower selling price.
        pos_frac = position / max(1, soft_limit)
        skew = self.PARAMS["option_inventory_skew"] * pos_frac * abs(pos_frac)
        buy_threshold = theo - edge - skew
        sell_threshold = theo + edge - skew

        best_bid, best_ask = self.best_bid_ask(order_depth)

        if best_ask is not None and best_ask <= buy_threshold and position < soft_limit:
            ask_volume = -order_depth.sell_orders[best_ask]
            qty = min(limit - position, soft_limit - position, ask_volume, max_trade)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                position += qty

        if best_bid is not None and best_bid >= sell_threshold and position > -soft_limit:
            bid_volume = order_depth.buy_orders[best_bid]
            qty = min(limit + position, soft_limit + position, bid_volume, max_trade)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                position -= qty

        return orders, position

    def hedge_vex_delta(
        self,
        state: TradingState,
        vex_orders: List[Order],
        simulated_positions: Dict[str, int],
        vex_mid: Optional[float],
    ) -> None:
        if vex_mid is None or "VELVETFRUIT_EXTRACT" not in state.order_depths:
            return

        total_option_delta = 0.0
        for product, K in self.VOUCHER_STRIKES.items():
            if product not in self.ACTIVE_VOUCHERS:
                continue
            pos = simulated_positions.get(product, state.position.get(product, 0))
            if pos == 0:
                continue
            delta = self.bs_delta(vex_mid, K, self.PARAMS["tte_days"], self.PARAMS["fair_iv"])
            total_option_delta += pos * delta

        hedge_fraction = self.PARAMS["vex_hedge_fraction"]
        target_vex_position = int(round(-hedge_fraction * total_option_delta))

        vex_limit = self.DELTA_PARAMS["VELVETFRUIT_EXTRACT"]["position_limit"]
        target_vex_position = max(-vex_limit, min(vex_limit, target_vex_position))

        current_vex_position = simulated_positions.get("VELVETFRUIT_EXTRACT", state.position.get("VELVETFRUIT_EXTRACT", 0))
        needed = target_vex_position - current_vex_position
        if needed == 0:
            return

        max_hedge_trade = self.PARAMS["vex_max_hedge_trade"]
        order_depth = state.order_depths["VELVETFRUIT_EXTRACT"]
        best_bid, best_ask = self.best_bid_ask(order_depth)

        if needed > 0 and best_ask is not None:
            ask_volume = -order_depth.sell_orders[best_ask]
            qty = min(needed, ask_volume, max_hedge_trade, vex_limit - current_vex_position)
            if qty > 0:
                vex_orders.append(Order("VELVETFRUIT_EXTRACT", best_ask, qty))
                simulated_positions["VELVETFRUIT_EXTRACT"] = current_vex_position + qty

        elif needed < 0 and best_bid is not None:
            bid_volume = order_depth.buy_orders[best_bid]
            qty = min(-needed, bid_volume, max_hedge_trade, vex_limit + current_vex_position)
            if qty > 0:
                vex_orders.append(Order("VELVETFRUIT_EXTRACT", best_bid, -qty))
                simulated_positions["VELVETFRUIT_EXTRACT"] = current_vex_position - qty

    # ---------- main entrypoint ----------
    def run(self, state: TradingState):
        try:
            saved = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}

        result: Dict[str, List[Order]] = {product: [] for product in state.order_depths}
        new_saved: Dict = {
            "fair_values": dict(saved.get("fair_values", {})),
            "mid_histories": dict(saved.get("mid_histories", {})),
        }
        simulated_positions: Dict[str, int] = dict(state.position)

        # 1. Delta-1 products first. VEX may also be used later as a hedge.
        for product in ("HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"):
            if product not in state.order_depths:
                continue
            orders, new_pos = self.trade_delta_product(
                product,
                state.order_depths[product],
                simulated_positions.get(product, 0),
                saved,
                new_saved,
            )
            result[product].extend(orders)
            simulated_positions[product] = new_pos

        # Need current VEX mid for option pricing.
        vex_mid = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            vex_mid = self.mid_price(state.order_depths["VELVETFRUIT_EXTRACT"])

        # 2. Voucher relative-value trading.
        if vex_mid is not None:
            for product in self.VOUCHER_STRIKES:
                if product not in state.order_depths:
                    continue
                orders, new_pos = self.trade_voucher(
                    product,
                    state.order_depths[product],
                    simulated_positions.get(product, 0),
                    vex_mid,
                )
                result[product].extend(orders)
                simulated_positions[product] = new_pos

            # 3. Hedge aggregate option delta using VEX.
            self.hedge_vex_delta(
                state,
                result.setdefault("VELVETFRUIT_EXTRACT", []),
                simulated_positions,
                vex_mid,
            )

        traderData = json.dumps(new_saved)
        conversions = 0
        return result, conversions, traderData
