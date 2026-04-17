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
            "default_anchor": 13000.0,
            "drift_per_timestamp": 0.0010,
            "passive_buy_edge": 6.0,
            "passive_sell_edge": 6.0,
            "aggressive_buy_edge": 8.0,
            "aggressive_sell_edge": 8.0,
            "max_skew": 3.0,
            "allow_short": True,
            "passive_order_size": 10,
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
        fair_values = saved_data.get("fair_values", {})
        prev_timestamp = saved_data.get("timestamp", state.timestamp)

        new_fair_values = dict(fair_values)
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
                result[product] = orders
                continue

            # -------------------------
            # FAIR VALUE
            # -------------------------
            if product == "INTARIAN_PEPPER_ROOT":
                fair_value = params["default_anchor"] + params["drift_per_timestamp"] * state.timestamp
            else:
                prev_fair = fair_values.get(product, params["default_fair"])
                drifted_prev_fair = prev_fair + params["drift_per_timestamp"] * dt
                alpha = params["ema_alpha"]
                fair_value = (1 - alpha) * drifted_prev_fair + alpha * mid_price

            new_fair_values[product] = fair_value

            # -------------------------
            # INVENTORY SKEW
            # -------------------------
            skew = (position / params["position_limit"]) * params["max_skew"]

            if product == "INTARIAN_PEPPER_ROOT":
                passive_bid_price = int(round(fair_value - params["passive_buy_edge"] - skew))
                passive_ask_price = int(round(fair_value + params["passive_sell_edge"] - skew))

                aggr_buy_threshold = fair_value - params["aggressive_buy_edge"] - skew
                aggr_sell_threshold = fair_value + params["aggressive_sell_edge"] - skew

                # Aggressive BUY
                if best_ask is not None:
                    ask_volume = -order_depth.sell_orders[best_ask]
                    max_can_buy = params["position_limit"] - position
                    if best_ask <= aggr_buy_threshold and max_can_buy > 0:
                        qty = min(max_can_buy, ask_volume)
                        if qty > 0:
                            orders.append(Order(product, best_ask, qty))
                            position += qty

                # Aggressive SELL
                if best_bid is not None:
                    bid_volume = order_depth.buy_orders[best_bid]
                    max_can_sell = params["position_limit"] + position
                    if best_bid >= aggr_sell_threshold and max_can_sell > 0:
                        qty = min(max_can_sell, bid_volume)
                        if qty > 0:
                            orders.append(Order(product, best_bid, -qty))
                            position -= qty

                # Passive BID
                max_can_buy = params["position_limit"] - position
                if max_can_buy > 0:
                    qty = min(params["passive_order_size"], max_can_buy)
                    if best_ask is None or passive_bid_price < best_ask:
                        orders.append(Order(product, passive_bid_price, qty))

                # Passive ASK
                max_can_sell = params["position_limit"] + position
                if max_can_sell > 0:
                    qty = min(params["passive_order_size"], max_can_sell)
                    if best_bid is None or passive_ask_price > best_bid:
                        orders.append(Order(product, passive_ask_price, -qty))

            else:
                buy_threshold = fair_value - params["buy_edge"] - skew
                sell_threshold = fair_value + params["sell_edge"] - skew

                if best_ask is not None:
                    ask_volume = -order_depth.sell_orders[best_ask]
                    max_can_buy = params["position_limit"] - position
                    if best_ask <= buy_threshold and max_can_buy > 0:
                        qty = min(max_can_buy, ask_volume)
                        if qty > 0:
                            orders.append(Order(product, best_ask, qty))
                            position += qty

                if best_bid is not None:
                    bid_volume = order_depth.buy_orders[best_bid]
                    max_can_sell = params["position_limit"] + position
                    if best_bid >= sell_threshold and max_can_sell > 0:
                        qty = min(max_can_sell, bid_volume)
                        if qty > 0:
                            orders.append(Order(product, best_bid, -qty))

            result[product] = orders

        traderData = json.dumps({
            "fair_values": new_fair_values,
            "timestamp": state.timestamp,
        })

        conversions = 0
        return result, conversions, traderData