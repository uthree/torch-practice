# utilites
def train2batch(x, y, batch_size=100):
    rx, ry = [], []
    for i in range(0, len(x), batch_size):
        rx.append(x[i:i+batch_size])
        ry.append(y[i:i+batch_size])
    return rx, ry

def test2batch(x, batch_size=100):
    rx = []
    for i in range(0, len(x), batch_size):
        rx.append(x[i:i+batch_size])
    return rx

# ロガー。 stringを詰め込んでファイルに保存する機能がある。
class Logger:
    def __init__(self, print_log=True):
        self.logs = []
        self.print_log = print_log
    
    def log(self, s:str, print_log=True):
        self.logs.append(s)
        if self.print_log and print_log:
            print(s)
    
    def save(self, path:str):
        with open(path, "w") as f:
            f.write("\n".join(self.logs))