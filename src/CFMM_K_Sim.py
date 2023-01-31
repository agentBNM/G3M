import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plt.style.use('_mpl-gallery')

def load_data():
    df = pd.read_csv("./code/data/xbtusd.csv");
    px = df['price'].to_numpy()
    return px

def swap(dx, dy):
    global alpha, k
    global x, y
    beta_inv = (1-alpha)/alpha
    beta = alpha/(1-alpha)
    
    if dy!=0:
        dx = - x/k * (1 - ( y / (y + k * dy) )**beta_inv )
        return np.abs(dx)
    else:
        dy = - y/k * (1 - ( x / (x + k * dx) )**beta )
        return np.abs(dy)

def pricing_curve(swap_amount_y):
    global alpha, k
    global x, y
    px_vec = np.zeros(swap_amount_y.shape)
    dx_vec = np.zeros(swap_amount_y.shape)
    dy_vec = np.zeros(swap_amount_y.shape)
    price_impact_vec = np.zeros(swap_amount_y.shape)
    j=0
    for amn in swap_amount_y:
        if amn<0:
            dx = -amn
            dy = swap(dx, 0)
            dx_vec[j] = -dx
            dy_vec[j] = dy
            px_vec[j] = dy/dx
            price_impact_vec[j] = (y-dy)/(x+dx)
        elif np.abs(amn)<1e-10:
            dx = 0
            dy = 0
            px_vec[j] = S[0]
            price_impact_vec[j] = S[0]
        else:
            dy = amn*S[0]
            dx = swap(0, dy)
            dx_vec[j] = dx
            dy_vec[j] = dy
            px_vec[j] = dy/dx
            price_impact_vec[j] = (y+dy)/(x-dx)
        j+=1
    return [dx_vec, dy_vec, px_vec, price_impact_vec]

if __name__=="__main__":
    
    S = load_data()
    x = 50
    y = x * S[0];
    swap_amount_y = np.arange(-10, 10, 0.25)
    
    # plot
    fig, ax = plt.subplots()
    
    alpha = 0.5
    k = 2
    [dx_vec, dy_vec, px_vec, price_impact] = pricing_curve(swap_amount_y)
    ax.plot(dx_vec, px_vec, 'b-', linewidth=2.0, label='alpha=0.5, k=2')
    ax.plot(dx_vec, price_impact, 'r-', linewidth=0.5, label='price impact: alpha=0.5, k=2')
    
    alpha = 0.5
    k = 1
    [dx_vec, dy_vec, px_vec, price_impact] = pricing_curve(swap_amount_y)
    ax.plot(dx_vec, px_vec, 'k--', linewidth=2.0, label='CPMM: alpha=0.5, k=1')
    ax.plot(dx_vec, price_impact, 'r--', linewidth=1.0, label='CPMM price impact')

    ax.hlines(S[0], np.min(dx_vec), np.max(dx_vec), linestyles=['dotted'], label="St")
    ax.legend()
    ax.set_title(f'Slippage. Initial AMM funds: x={x:.2f}, y={y:.2f}')
    ax.set_xlabel("#BTC trade size")
    ax.set_ylabel("Price")
    plt.show()

    dy = 13771
    dx = swap(0, dy)
    S_new = y/x
    print(dy/dx)
    print(S_new)
    print("fee=", (dy/dx-S[0])/S[0]-1)
    

