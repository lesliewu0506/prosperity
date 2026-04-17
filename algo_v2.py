from datamodel import OrderDepth, TradingState, Order
from typing import List
import json


class Trader:
    PARAMS = {
        "ASH_COATED_OSMIUM": {
            "position_limit": 80,
            "default_fair": 10000.0,
            "ema_alpha": 0.15,
            "buy_edge": 1,
            "sell_edge": 1,
            "max_skew": 2.0,
            "drift_per_timestamp": 0.0,
            "allow_short": True,
        },
        "INTARIAN_PEPPER_ROOT": {
            "position_limit": 80,
            "default_anchor": 12000.0,
            "drift_per_timestamp": 0.0010,
            "buy_edge": 1.5,
            "sell_edge": 1.5,
            "max_skew": 3.0,
            "allow_short": False,
        },
    }

    def bid(self):
        return 15

    def run(self, state: TradingState):
        try:
            saved_data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved_data = {}

        result = {}

        # Persistent state
        fair_values = saved_data.get("fair_values", {})
        anchors = saved_data.get("anchors", {})
        prev_timestamp = saved_data.get("timestamp", state.timestamp)

        new_fair_values = dict(fair_values)
        new_anchors = dict(anchors)

        dt = state.timestamp - prev_timestamp

        for product in state.order_depths:
            orders: List[Order] = []

            if product not in self.PARAMS:
                result[product] = orders
                continue

            params = self.PARAMS[product]
            order_depth: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)

            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

            if best_bid is not None and best_ask is not None:
                mid_price = (best_bid + best_ask) / 2.0
            elif best_bid is not None:
                mid_price = float(best_bid)
            elif best_ask is not None:
                mid_price = float(best_ask)
            else:
                # No market data at all
                if product == "INTARIAN_PEPPER_ROOT":
                    anchor = anchors.get(product, params["default_anchor"])
                    fair_value = anchor + params["drift_per_timestamp"] * state.timestamp
                else:
                    fair_value = fair_values.get(product, params["default_fair"])
                result[product] = orders
                new_fair_values[product] = fair_value
                continue

            # -------------------------
            # FAIR VALUE CALCULATION
            # -------------------------

            if product == "INTARIAN_PEPPER_ROOT":
                # Deterministic trend model:
                # fair = anchor + drift * timestamp
                # Infer anchor from first observed mid if not yet known
                drift = params["drift_per_timestamp"]

                if product not in anchors:
                    inferred_anchor = mid_price - drift * state.timestamp
                    new_anchors[product] = inferred_anchor
                anchor = new_anchors.get(product, params["default_anchor"])

                fair_value = anchor + drift * state.timestamp

            else:
                # ACO keeps EMA-based fair
                prev_fair = fair_values.get(product, params["default_fair"])
                drifted_prev_fair = prev_fair + params["drift_per_timestamp"] * dt
                alpha = params["ema_alpha"]
                fair_value = (1 - alpha) * drifted_prev_fair + alpha * mid_price

            new_fair_values[product] = fair_value

            # -------------------------
            # INVENTORY SKEW
            # -------------------------
            skew = (position / params["position_limit"]) * params["max_skew"]

            buy_threshold = fair_value - params["buy_edge"] - skew
            sell_threshold = fair_value + params["sell_edge"] - skew

            # -------------------------
            # BUY LOGIC
            # -------------------------
            if best_ask is not None:
                best_ask_volume = order_depth.sell_orders[best_ask]  # negative in sell book
                available_to_buy = -best_ask_volume
                max_can_buy = params["position_limit"] - position

                if best_ask <= buy_threshold and max_can_buy > 0:
                    buy_qty = min(max_can_buy, available_to_buy)
                    if buy_qty > 0:
                        orders.append(Order(product, best_ask, buy_qty))
                        position += buy_qty  # local update for sell sizing

            # -------------------------
            # SELL LOGIC
            # -------------------------
            if best_bid is not None:
                best_bid_volume = order_depth.buy_orders[best_bid]

                if params["allow_short"]:
                    max_can_sell = params["position_limit"] + position
                else:
                    max_can_sell = max(position, 0)

                if best_bid >= sell_threshold and max_can_sell > 0:
                    sell_qty = min(max_can_sell, best_bid_volume)
                    if sell_qty > 0:
                        orders.append(Order(product, best_bid, -sell_qty))

            result[product] = orders

        traderData = json.dumps({
            "fair_values": new_fair_values,
            "anchors": new_anchors,
            "timestamp": state.timestamp,
        })

        conversions = 0
        return result, conversions, traderData