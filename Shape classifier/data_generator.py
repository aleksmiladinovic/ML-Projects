from matplotlib import pyplot as plt
import random
import math
import numpy as np
import pickle


N = 5000
shapes = ['rectangle', 'circle', 'triangle']

with open("data description.aca","wb") as fd, open("data.aca","wb") as f:
    img_size = [26, 35, 4]
    pickle.dump(N,f)
    pickle.dump(img_size,f)
    for i in range(N):
        img_type = random.randint(0, 2)
        n = random.randint(550, 700)
        x = []
        y = []
        if img_type == 0:
            dx = random.randrange(10, 15)
            dy = random.randrange(10, 15)
            a = random.randrange(52, 60)
            b = random.randrange(52, 60)
            for j in range(n):
                x.append(random.uniform(dx, dx+a))
                y.append(random.uniform(dy, dy+b))
        elif img_type == 1:
            cx = random.randrange(37, 43)
            cy = random.randrange(37, 43)
            R = random.randrange(26, 30)
            for j in range(n):
                r = random.uniform(0, R)
                theta = random.uniform(0, 2*math.pi)
                x.append(cx+r*math.cos(theta))
                y.append(cy+r*math.sin(theta))
        else:
            dx = random.randrange(12, 15)
            dy = random.randrange(12, 15)
            d = random.randrange(50, 55)
            ah = random.uniform(float(d)/3,2*float(d)/3)
            h = random.randrange(55, 60)
            for j in range(n):
                xn = random.uniform(dx, dx+d)
                if xn <= dx+ah:
                    k = float(h)/ah
                    n = dy-k*dx
                else:
                    k = float(h)/(ah-d)
                    n = dy-k*(dx+d)
                yn = random.uniform(dy, k * xn + n)
                x.append(xn)
                y.append(yn)
        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.xlim(0,80)
        plt.ylim(0,80)
        ax.scatter(x, y)
        ax.axis('off')
        fig.savefig('pic.png', bbox_inches='tight')
        plt.close(fig)

        img = plt.imread('pic.png')
        gap = 15
        img_new = np.asarray([[[x for x in list(img[i,j,:])] for j in range(0,img.shape[1],gap)] for i in range(0,img.shape[0],gap)])

        pickle.dump(img_new.flatten(), f)
        pickle.dump(shapes[img_type], fd)
    f.close()
    fd.close()
