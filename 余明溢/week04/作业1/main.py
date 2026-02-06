from model.bert import model_for_bert
if __name__ == '__main__':
    print(model_for_bert([
        "你好，支付失败了怎么办",
        "我的订单怎么还没发货",
        "这个商品可以退货吗",
        "有没有优惠券可以用",
        "账号密码忘记了怎么找回",
        "我要投诉客服态度不好",
        "这个手机多少钱"
    ]))