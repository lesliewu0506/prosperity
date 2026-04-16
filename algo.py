from datamodel import OrderDepth, UserId, TradingState, Order
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
            "max_skew": 2,
            "drift_per_timestamp": 0.0,
            "allow_short": True,
        },
        "INTARIAN_PEPPER_ROOT": {
            "position_limit": 80,
            "default_fair": 10000.0,
            "ema_alpha": 0.10,
            "buy_edge": 2,
            "sell_edge": 6,
            "max_skew": 4,
            "drift_per_timestamp": 0.0010,
            "allow_short": False,
        },
    }

    def bid(self):
        return 15

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        try:
            saved_data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved_data = {}

        fair_values = saved_data.get("fair_values", {})
        prev_timestamp = saved_data.get("timestamp", state.timestamp)

        result = {}
        new_fair_values = dict(fair_values)

        dt = state.timestamp - prev_timestamp

        for product in state.order_depths:
            orders: List[Order] = []

            if product not in self.PARAMS:
                result[product] = []
                continue

            params = self.PARAMS[product]
            order_depth: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)

            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

            prev_fair = fair_values.get(product, params["default_fair"])

            # Mid price
            if best_bid is not None and best_ask is not None:
                mid_price = (best_bid + best_ask) / 2
            elif best_bid is not None:
                mid_price = float(best_bid)
            elif best_ask is not None:
                mid_price = float(best_ask)
            else:
                mid_price = prev_fair

            # Drift-aware fair value
            drifted_prev_fair = prev_fair + params["drift_per_timestamp"] * dt
            fair_value = (
                (1 - params["ema_alpha"]) * drifted_prev_fair
                + params["ema_alpha"] * mid_price
            )
            new_fair_values[product] = fair_value

            # Inventory skew
            skew = round((position / params["position_limit"]) * params["max_skew"])

            buy_threshold = fair_value - params["buy_edge"] - skew
            sell_threshold = fair_value + params["sell_edge"] - skew

            # BUY LOGIC
            if best_ask is not None:
                best_ask_volume = order_depth.sell_orders[best_ask] 
                available_to_buy = -best_ask_volume
                max_can_buy = params["position_limit"] - position

                if best_ask < buy_threshold and max_can_buy > 0:
                    buy_qty = min(max_can_buy, available_to_buy)
                    if buy_qty > 0:
                        print("BUY", str(buy_qty) + "x", best_ask)
                        orders.append(Order(product, best_ask, buy_qty))
                        position += buy_qty

            # SELL LOGIC
            if best_bid is not None:
                best_bid_volume = order_depth.buy_orders[best_bid]

                if params["allow_short"]:
                    max_can_sell = params["position_limit"] + position
                else:
                    max_can_sell = max(position, 0) 

                if best_bid > sell_threshold and max_can_sell > 0:
                    sell_qty = min(max_can_sell, best_bid_volume)
                    if sell_qty > 0:
                        print("SELL", str(sell_qty) + "x", best_bid)
                        orders.append(Order(product, best_bid, -sell_qty))

            result[product] = orders

        traderData = json.dumps({
            "fair_values": new_fair_values,
            "timestamp": state.timestamp
        })
        conversions = 0
        return result, conversions, traderData