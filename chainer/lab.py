from train import RecursiveNet, traverse, PredictNet
import chainer.computational_graph as c

sentence = '[["How @ 0", [["can @ 468", ["I @ 33", ["increase @ 222", [[["the @ 2", "speed @ 1002"], ["of @ 76", ["my @ 106", ["internet @ 1174", "connection @ 3430"]]]], ["while @ 1133", ["using @ 575", ["a @ 93", "VPN @ 2615"]]]]]]], "? @ 10"]], ["How @ 0", [["can @ 468", [["Internet @ 1176", "speed @ 1002"], ["be @ 121", ["increased @ 5998", [["by @ 123", "hacking @ 1776"], ["through @ 514", "DNS @ 26222"]]]]]], "? @ 10"]], "0", "2"]'

vocab_size = 153451
embedding_size = 30

predict = PredictNet(vocab_size, embedding_size)

pred, id = predict(sentence)
print(pred, id)


g = c.build_computational_graph([pred])
with open('./graph', 'w') as f:
    f.write(g.dump())