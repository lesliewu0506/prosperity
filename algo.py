from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import json

class Trader:

    POSITION_LIMIT = 80

    # Fair value model
    DEFAULT_FAIR_VALUE = 10000.0
    EMA_ALPHA = 0.15

    # Trading edge around fair value
    BASE_EDGE = 1

    MAX_SKEW = 2

    def bid(self):
        return 15
    
    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""

        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        # Read previous fair value from traderData
        try:
            saved_data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved_data = {}

        prev_fair_value = saved_data.get("fair_value", self.DEFAULT_FAIR_VALUE)
        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            orders: List[Order] = []
            if product != "ASH_COATED_OSMIUM":
                result[product] = []
                continue

            else:
                order_depth: OrderDepth = state.order_depths[product]
                position = state.position.get(product, 0)

                best_bid = (max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None)
                best_ask = (min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None)

                # Fair Value Calculation
                if best_bid is not None and best_ask is not None:
                    mid_price = (best_bid + best_ask) / 2
                elif best_bid is not None:
                    mid_price = float(best_bid)
                elif best_ask is not None:
                    mid_price = float(best_ask)
                else:
                    mid_price = prev_fair_value
                
                fair_value = (1 - self.EMA_ALPHA) * prev_fair_value + self.EMA_ALPHA * mid_price

                # Inventory Skew Calculation
                skew = round((position / self.POSITION_LIMIT) * self.MAX_SKEW)

                buy_threshold = fair_value - self.BASE_EDGE - skew
                sell_threshold = fair_value + self.BASE_EDGE - skew

                # Buy Algorithm
                if len(order_depth.sell_orders) != 0:
                    best_ask_volume = order_depth.sell_orders[best_ask] 
                    available_to_buy = -best_ask_volume
                    max_can_buy = self.POSITION_LIMIT - position

                    if best_ask < buy_threshold and max_can_buy > 0:
                        buy_qty = min(max_can_buy, available_to_buy)
                        print("BUY", str(buy_qty) + "x", best_ask)
                        orders.append(Order(product, best_ask, buy_qty))

                # Sell Algorithm
                if len(order_depth.buy_orders) != 0:
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    max_can_sell = self.POSITION_LIMIT + position

                    if best_bid > sell_threshold and max_can_sell > 0:
                        sell_qty = min(max_can_sell, best_bid_volume)
                        print("SELL", str(sell_qty) + "x", best_bid)
                        orders.append(Order(product, best_bid, -sell_qty))
                
                result[product] = orders
    
        traderData = ""  # No state needed - we check position directly
        conversions = 0
        return result, conversions, traderData