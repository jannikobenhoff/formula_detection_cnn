import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_json(requests.get("http://127.0.0.1:7777/").text)
    print(df["values"])
    response = requests.get("http://127.0.0.1:7777")
    print(response.text[2:-3].split(";"))
    img = []
    for i in df["values"].split(";"):#response.text[2:-2].split(";"):
        #i = i[1:-1]
        print(i)
        if i.isdigit():
            img.append(int(i))

    img = np.array(img).reshape(28, 28)
    print(len(img))
    plt.imshow(img, cmap="gray")
    plt.show()