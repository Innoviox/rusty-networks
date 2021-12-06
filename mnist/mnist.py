def print_img(image):
    for i in image:
        for j in i:
            s = '1' if ord(j) < 127 else ' '
            print(s, end='')
        print()

with open("train-labels-idx1-ubyte", "rb") as g:
    g.read(8)
    with open("train-images-idx3-ubyte", "rb") as f:
        f.read(16)
        for i in range(60000):
            img = [[0 for _ in range(28)] for _ in range(28)]
            for j in range(784):
                img[j % 28][j // 28] = f.read(1)

            print_img(img)
            print(ord(g.read(1)))
            input()
            
