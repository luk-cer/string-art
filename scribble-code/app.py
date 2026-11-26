from PIL import Image
import numpy as np 
from scipy.ndimage import sobel
from matplotlib import pyplot as plt 

def rejection_sampler(prob_mask, N):
    samples = []
    while len(samples)<N:
        idx = np.random.randint(len(prob_mask))
        if idx in samples:
            continue
        if np.random.uniform(0,1)<prob_mask[idx]:
            samples.append(idx)
    return sorted(samples)

def rescale(X):
    X -= X.min()
    X /= X.max()
    return X

image = Image.open('image.jpg')
print('original')
plt.imshow(image)
plt.tight_layout()
plt.axis('off')
plt.show() 

image = image.convert('L')
pix = np.array(image, dtype=float)
pix /=255

edges_x = sobel(pix,0)
edges_y = sobel(pix,1)
magnitude = np.sqrt(edges_x**2 + edges_y**2)
magnitude = rescale(magnitude)
magnitude[magnitude<0.05] = 0 
edges_x[magnitude==0] = 0
edges_y[magnitude==0] = 0

print('gradient magnitude')
plt.imshow(magnitude)
plt.tight_layout()
plt.axis('off')
plt.show()


ipix = 1 - pix

threshold = 0.2
sampling_pct = 0.1
prob_mask = ipix 
min_prob = ipix.min() + (ipix.max() - ipix.min())* threshold
prob_mask[prob_mask<min_prob] = 0 

N = int(np.sum(prob_mask>0) * sampling_pct)
#samples = rejection_sampler(prob_mask.flatten(), N)
#points = np.unravel_index(samples, prob_mask.shape)

from scipy.sparse import coo_matrix

values = np.ones(len(magnitude[0]))
#sample_image = coo_matrix((values,points), shape=pix.shape)

print('probmask')
plt.imshow(prob_mask)
plt.tight_layout()
plt.axis('off')
plt.show()

#print('sample image')
#plt.imshow(sample_image.todense())
#plt.tight_layout()
#plt.axis('off')
#plt.show()

import numpy as np
scribble = np.zeros(shape=pix.shape)
d=3
weights = np.ones(shape=(2*d+1, 2*d+1))
for i in range(-d,d+1):
    for j in range(-d, d+1):
        w = 1/ (1+np.exp(-np.sqrt(i*i+j*j)))
        weights[i+d,j+d]= w
weights = weights / np.sum(weights)

print('weights')
plt.imshow(weights)
plt.tight_layout()
plt.axis('off')
plt.show()

def gradient_normal(x,y, g_x, g_y, right=True, d=1, weights=None):
    if weights is None:
        weights = np.ones(shape=(2*d+1, 2*d+1)) 
    weights = weights / weights.sum()
    x_size, y_size = g_x.shape
    x_min = max(x-d, 0)
    x_max = min(x+d+1, x_size)
    y_min = max(y-d, 0)
    y_max = min(y+d+1, y_size)
    wx_min = abs(x - d) if x-d < 0 else 0  
    wx_max = d - x + x_size if x + d >= x_size else 2*d+1
    wy_min = abs(y - d) if y - d < 0 else 0 
    wy_max = d - y + y_size if y + d > y_size-1 else 2*d+1
    s_x= g_x[x_min:x_max, y_min:y_max]
    s_y= g_y[x_min:x_max, y_min:y_max]
    weights = weights[wx_min:wx_max, wy_min:wy_max]
    try:
        if right:
            a = -np.sum(s_y * weights),np.sum(s_x * weights)
        else:
            a = np.sum(s_y * weights),-np.sum(s_x * weights) 
        return a
    except:
        print(weights.shape, s_y.shape, s_x.shape)
        print(x - d , x+d+1, x_size, y-d, y+d+1, y_size)
        print(x,d,x_size, y, d, y_size)
        print(wx_min, wx_max, wy_min, wy_max)
        raise Exception("aaa")


magnitude /= magnitude.sum()
def pick_point(magnitude, prob=True):
    if prob:
        p = magnitude / magnitude.sum()
        i = np.random.choice(np.arange(p.size), p=p.ravel())
        sample = np.unravel_index(i, p.shape)
    else:
        arg_max = np.where(magnitude==np.max(magnitude))
        sample = int(arg_max[0][0]), int(arg_max[1][0])
    return sample

def draw_line(image, s_x, s_y, e_x, e_y):
    steps =max(abs(s_x-e_x), abs(s_y-e_y))
    for i in range(steps):
        x = int(s_x + i/steps * (e_x - s_x))
        y = int(s_y + i/steps * (e_y - s_y))
        image[x,y] = 1
    return image

def sample_next_point(s_x, s_y, d_x, d_y, density_map, d=10):
    shape =density_map.shape
    points=[]
    while len(points)==0:
        min_x = max(0,s_x+d_x-d)
        max_x = min(shape[0],s_x+d_x+d+1)
        min_y = max(0,s_y+d_y-d)
        max_y = min(shape[0],s_y+d_y+d+1)
        points = np.where(density_map[min_x:max_x,min_y:max_y]>0)
        d += 1
        h = np.sqrt(d_x*d_x + d_y*d_y)
        d_x = d_x / h 
    len(points)>0

def delete_region(a, s_x,s_y, d=5):
    shape = a.shape
    min_x = max(0,s_x-d)
    max_x = min(shape[0],s_x+d+1)
    min_y = max(0,s_y-d)
    max_y = min(shape[0],s_y+d+1)
    a[min_x:max_x, min_y:max_y]=0 
    return a 

def trace_gradient(px,py,gx,gy,intensity, right=True, weights=weights, max_steps=1000,d=5):
    line =[(px,py)]
    sintensity = intensity[px,py]
    step=0
    cx = px 
    cy = py 
    while (-1< px <pix.shape[0]-1 and -1< py <pix.shape[1]-1 
           and len(line) == len(set(line))
           and abs(intensity[cx, cy]-sintensity*0.7)<0.7
           and step< max_steps):
        #print('intensity',intensity[cx, cy], sintensity)
        cx, cy = line[-1]
        g_x, g_y = gradient_normal(cx, cy, 
                        g_x=gx, 
                        g_y=gy, 
                        d=d, 
                        weights=weights,
                        right=right)
        if max(abs(g_x), abs(g_y))==abs(g_x):
            if g_x ==0:
                break
            g_y = np.sign(g_y) * abs(g_y / g_x) 
            g_x = np.sign(g_x)
        else:
            if g_y ==0:
                break
            g_x = np.sign(g_x) * abs(g_x / g_y)
            g_y = np.sign(g_y)
        px = px + g_x
        py = py + g_y
        if px <0 or py<0:
            break
        cx, cy = int(px), int(py)
        #print('intensity',intensity[cx, cy], sintensity)
        line.append((cx,cy))
        step += 1
    return line

def plot_lines(lines, shape):
    result = np.zeros(shape=shape)
    for line in lines:
        if len (line)>1:
            try:
                values = np.ones( len(line))
                x_line, y_line = zip(*line)
                line_image = coo_matrix((values, (x_line, y_line)), shape=pix.shape).todense()
            except Exception as e:
                print(len(line), e)
            result += line_image
    result[result >0] = 1.0
    print('line image')
    plt.imshow(result, cmap='Greys')
    plt.tight_layout()
    plt.axis('off')
    plt.show()

def gradient_clean_up(line, cleanup_pairs, ipix, max_d=1):
    result = []
    for x, y in line:
        intensity = ipix[x,y]
        d = int((max_d -1)*(1.0-intensity)+1)
        for item, value in cleanup_pairs:
            item[x-d:x+d+1,y-d:y+d+1] = value
    result = [itm[0] for itm in cleanup_pairs]
    return result

def calc_intensity(intensities, x ,y, d):
    mx, my = intensities.shape
    xdp = max(min(x+d+1, mx),1)
    xdn = max(x-d,0)
    ydp = max(min(y+d+1, my),1)
    ydn = max(y-d,0)
    u = 0 
    try:
        ixpyp = int(u + (1-intensities[x:xdp,y:ydp].max())*(d-u))
        ixpyn = int(u + (1-intensities[x:xdp,ydn:y+1].max())*(d-u))
        ixnyp = int(u + (1-intensities[xdn:x+1,y:ydp].max())*(d-u))
        ixnyn = int(u + (1-intensities[xdn:x+1,ydn:y+1].max())*(d-u))
    except Exception as e:
        print('ci', x,y,d)
        print(intensities[x:x+d+1,y:y+d+1].shape)
        print(intensities[x:x+d+1,y-d:y+1].shape)
        print(intensities[x-d:x+1,y:y+d+1].shape)
        print(intensities[x-d:x+1,y-d:y+1].shape)
        raise e
    return ixpyp, ixpyn, ixnyp, ixnyn

def gradient_clean_up2(line, cleanup_pairs, ipix, max_d=1):
    result = []
    for x, y in line:
        pp, pn, np, nn =calc_intensity(ipix, x,y, max_d)
        for item, value in cleanup_pairs:
            try:
                item[x:x+pp+1,y:y+pp+1] = value
                item[x:x+pn+1,y-pn:y+1] = value
                item[x-np:x+1,y:y+np+1] = value
                item[x-nn:x+1,y-nn:y+1] = value
            except:
                print(x,y,d, pp, pn, np, nn)
    result = [itm[0] for itm in cleanup_pairs]
    return result

i = 0
gx = edges_x
gy = edges_y
points = np.prod(magnitude.shape)
lines = []
d_clean = (d * 3) 
for i in range(10000):
    if np.sum(magnitude>0) / points < 10**-3:
        break
    print(f'line {i}')
    try:
        px,py = pick_point(magnitude, True)
        if ipix[px][py] < 0.01:
            gx, gy, magnitude = gradient_clean_up([(px,py)], ((gx,0), (gy,0), (magnitude,0)), ipix,3)
            continue
        lines.append(trace_gradient(px, py, gx, gy, ipix, right=True, weights=weights,d=d, max_steps=10000))
        lines.append(trace_gradient(px, py, gx, gy, ipix, right=False, weights=weights,d=d, max_steps=10000))
        gx, gy, magnitude = gradient_clean_up2(lines[-1], ((gx,0), (gy,0), (magnitude,0)), ipix, d*5)
        gx, gy, magnitude = gradient_clean_up2(lines[-1], ((gx,0), (gy,0), (magnitude,0)), ipix, d*5)
        if i%500 ==0 :
            plt.imshow(magnitude)
            plt.show()
    except Exception as e:
        print(e)
        break

len(lines)
plot_lines(lines, pix.shape)

new_lines = [line for line in lines if len(line)>50]
print('remaining lines', len(new_lines))
plot_lines(new_lines, pix.shape)

import math

def gradient(x,y, g_x, g_y, angle=90, d=1, weights=None):
    rad = angle * np.pi / 180.0
    if weights is None:
        weights = np.ones(shape=(2*d+1, 2*d+1)) 
    weights = weights / weights.sum()
    x_size, y_size = g_x.shape
    x_min = max(x-d, 0)
    x_max = min(x+d+1, x_size)
    y_min = max(y-d, 0)
    y_max = min(y+d+1, y_size)
    wx_min = abs(x - d) if x-d < 0 else 0  
    wx_max = d - x + x_size if x + d >= x_size else 2*d+1
    wy_min = abs(y - d) if y - d < 0 else 0 
    wy_max = d - y + y_size if y + d > y_size-1 else 2*d+1
    s_x= g_x[x_min:x_max, y_min:y_max]
    s_y= g_y[x_min:x_max, y_min:y_max]
    weights = weights[wx_min:wx_max, wy_min:wy_max]
    try:
        if right:
            a = -np.sum(s_y * weights),np.sum(s_x * weights)
        else:
            a = np.sum(s_y * weights),-np.sum(s_x * weights) 
        return a
    except:
        print(weights.shape, s_y.shape, s_x.shape)
        print(x - d , x+d+1, x_size, y-d, y+d+1, y_size)
        print(x,d,x_size, y, d, y_size)
        print(wx_min, wx_max, wy_min, wy_max)
        raise Exception("aaa")


def trace(px,py,gx,gy,intensity, angle=90, weights=weights, max_steps=1000,d=5):
    line =[(px,py)]
    sintensity = intensity[px,py]
    step=0
    cx = px 
    cy = py 
    while (-1< px <pix.shape[0]-1 and -1< py <pix.shape[1]-1 
           and len(line) == len(set(line))
           and abs(intensity[cx, cy]-sintensity*0.7)<0.7
           and step< max_steps):
        #print('intensity',intensity[cx, cy], sintensity)
        cx, cy = line[-1]
        g_x, g_y = gradient_normal(cx, cy, 
                        g_x=gx, 
                        g_y=gy, 
                        d=d, 
                        weights=weights,
                        right=right)
        if max(abs(g_x), abs(g_y))==abs(g_x):
            if g_x ==0:
                break
            g_y = np.sign(g_y) * abs(g_y / g_x) 
            g_x = np.sign(g_x)
        else:
            if g_y ==0:
                break
            g_x = np.sign(g_x) * abs(g_x / g_y)
            g_y = np.sign(g_y)
        px = px + g_x
        py = py + g_y
        if px <0 or py<0:
            break
        cx, cy = int(px), int(py)
        #print('intensity',intensity[cx, cy], sintensity)
        line.append((cx,cy))
        step += 1
    return line


