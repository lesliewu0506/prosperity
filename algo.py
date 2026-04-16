from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List

class Trader:

    POSITION_LIMIT = 80
    FAIR_PRICE_OSMIUM = 10000
    BUY_PRICE_OSMIUM = 9998
    SELL_PRICE_OSMIUM = 10010

    def bid(self):
        return 15
    
    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""

        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

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


                # Buy Algorithm
                if len(order_depth.sell_orders) != 0:
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    available_to_buy = -best_ask_amount
                    max_can_buy = self.POSITION_LIMIT - position

                    if int(best_ask) < self.BUY_PRICE_OSMIUM and max_can_buy > 0:
                        buy_qty = min(max_can_buy, available_to_buy)
                        print("BUY", str(buy_qty) + "x", best_ask)
                        orders.append(Order(product, best_ask, buy_qty))

                # Sell Algorithm
                if len(order_depth.buy_orders) != 0:
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    max_can_sell = self.POSITION_LIMIT + position

                    if int(best_bid) > self.SELL_PRICE_OSMIUM and max_can_sell > 0:
                        sell_qty = min(max_can_sell, best_bid_amount)
                        print("SELL", str(sell_qty) + "x", best_bid)
                        orders.append(Order(product, best_bid, -sell_qty))
                
                result[product] = orders
    
        traderData = ""  # No state needed - we check position directly
        conversions = 0
        return result, conversions, traderData