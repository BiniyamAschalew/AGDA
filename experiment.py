from config.configurator import configs

def test1(iter=1):
    for i in range(iter):
        configs["a"] = configs.get("a", 0) + 1
        print(configs["a"])