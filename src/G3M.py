import numpy as np
import pandas as pd
import datetime
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize_scalar
import glob

def H(x):
    return 0 if x<=0 else 1

def load_data(N=-1):
    df = pd.read_csv("./data/xbtusd.csv");
    px = df['price'].to_numpy()
    ts = list(map(datetime.datetime.fromisoformat, df['time']))
    if N>0:
        return [px[0:N], ts[0:N]]
    return [px, ts]

def exchangeRate():
    global x, y, alpha
    beta = alpha/(1-alpha)
    return beta * y/x

def get_dy(dx, x, y, tau, beta):
    return -y * ( 1 - ( x/( x+(1-tau)*dx ) )**beta ) 

def get_dx(dy, x, y, tau, beta):
    return -x * ( 1 - ( y/( y+(1-tau)*dy ) )**(1/beta) ) 

def swap(_dx, _dy, tau):
    global alpha
    global x, y
    beta = alpha/(1-alpha)
    
    if _dx>0:
        dy = get_dy(_dx, x, y, tau, beta)
        dx = _dx
    else:
        dx = get_dx(_dy, x, y, tau, beta)
        dy = _dy

    a = ( x+(1-tau*H(dx)) * dx ) ** alpha * ( y+(1-tau*H(dy)) * dy ) ** (1-alpha) 
    b = x**alpha * y**(1-alpha)
    assert(np.abs(a-b) < 1e-10)
    # adjust reserves
    x = x+dx
    y = y+dy
    return (dx, dy)

def bid_ask(tau):
    global alpha
    global x, y
    beta = alpha/(1-alpha)
    a = 1/(1-tau) * beta * y/x
    b = (1-tau) * beta * y/x
    return [b, a]

def largest_small_buyX(St, tau):
    global alpha
    global x, y
    # pre-condition: no arbitrage
    [b,a] = bid_ask(tau)
    assert(St>=b and St<=a)

    beta = alpha/(1-alpha)
    def sq_error(dy):
        dx = get_dx(dy, x, y, tau, beta)
        return (beta * (y+dy) / (x+dx) * (1-tau) - St )**2
    res = minimize_scalar(sq_error,bounds=(0,1e9), method='bounded')
    dy = res.x
    return [get_dx(dy, x, y, tau, beta), dy]

def largest_small_sellX(St, tau):
    global alpha
    global x, y
    # pre-condition: no arbitrage
    [b,a] = bid_ask(tau)
    assert(St>=b and St<=a)

    beta = alpha/(1-alpha)
    def sq_error(dx):
        dy = get_dy(dx, x, y, tau, beta)
        # new bid price = St
        return (beta * (y+dy) / (x+dx)  - St * (1-tau))**2
    res = minimize_scalar(sq_error,bounds=(0,2), method='bounded')
    dx = res.x
    return [dx, get_dy(dx, x, y, tau, beta)]

def buy_arbitrage(St, tau):
    global alpha
    global x, y
    #S[t]>ask
    # pre-condition: arbitrage
    [_,a] = bid_ask(tau)
    assert(St>a)

    def sq_error(dy):
        dx = get_dx(dy, x, y, tau, beta)
        return (beta * (y+(1-tau) * dy) / (x+dx)  - St * (1-tau))**2
    res = minimize_scalar(sq_error,bounds=(0,1e9), method='bounded')
    dy = res.x
    return [get_dx(dy, x, y, tau, beta), dy]

def sell_arbitrage(St, tau):
    global alpha
    global x, y
    #S[t]<bid
    # pre-condition: arbitrage
    [b,_] = bid_ask(tau)
    assert(St<b)

    def sq_error(dx):
        dy = get_dy(dx, x, y, tau, beta)
        # new bid price = St
        return (beta * (y+dy) / (x+(1-tau) * dx)  - St / (1-tau))**2
    res = minimize_scalar(sq_error,bounds=(0,2), method='bounded')
    dx = res.x
    return [dx, get_dy(dx, x, y, tau, beta)]

def exec_arbitrage_trade(St, tau):
    global alpha
    global x, y
    [bid, ask] = bid_ask(tau)
    b=bid
    a=ask
    k=0
    num_arb = 0
    while St<b or St>a:
        if b>St:
            [dx, dy] = sell_arbitrage(St, tau)
            swap(dx, 0, tau)
            num_arb+=1
        elif a<St:
            [dx, dy] = buy_arbitrage(St, tau)
            swap(0, dy, tau)
            num_arb+=1
        [b,a] = bid_ask(tau)
        k=k+1
    #if k>1:
    #    print("num arbitrage trades=",k)
    assert(St>=b and St<=a)
    return num_arb

def sim(prob_small, prob_trade, y_start, tau):
    global alpha
    global x, y
    global S
    eps = 1e-3
    prob_buy = 0.5
    
    x = beta * y_start/S[0]
    y = y_start
    x0 = x
    #[X, Y, px, total_arb_trades, total_large_trades, total_small_trades]
    df = pd.DataFrame(columns=["S", "X", "Y", "px", "Vbh", "Vlp", "IL", "Phi", "ArbTrades", "LargeTrades", "SmallTrades"])
    df["S"] = S
    Phi = 0
    total_arb_trades = 0
    total_large_trades = 0
    total_small_trades = 0
    S_ = S[1] #hack for first element in Phi
    x_ = x
    y_ = y
    for t in range(S.shape[0]):
        # optimal arbitrage trade if outside bounds
        trades = exec_arbitrage_trade(S[t], tau)
        total_arb_trades += trades

        if trades==0:
            # continuous model assumes that S moves after each
            # arb trade, small trade, large trade+arb
            if np.random.uniform(0,1) < prob_trade:
                trades=1
                if np.random.uniform(0,1) < prob_small:
                    # perform a small trade
                    if np.random.uniform(0,1)<prob_buy:
                        # buy
                        resXY = largest_small_buyX(S[t], tau)
                        trade = resXY[1]-eps
                        if trade>0:
                            swap(0, resXY[1]-eps, tau)
                            total_small_trades += 1
                    else:
                        resXY = largest_small_sellX(S[t], tau)
                        trade = resXY[0]-eps/S[t]
                        if trade>0:
                            swap(resXY[0]-eps/S[t], 0, tau)
                            total_small_trades += 1
                else:
                    # perform large trade
                    if np.random.uniform(0,1)<prob_buy:
                        # buy
                        resXY = largest_small_buyX(S[t], tau)
                        swap(0, resXY[1]*2, tau)
                        total_large_trades +=1
                    else:
                        # sell: swap x for y
                        resXY = largest_small_sellX(S[t], tau)
                        swap(resXY[0]*2, 1, tau)
                        total_large_trades +=1
                # optimal arbitrage trade if outside bounds
                total_arb_trades += exec_arbitrage_trade(S[t], tau)
        # record funds
        #updatePhi
        Phi = Phi + (x0 - x) * (S[t] - S_)
        vbh = y0 + S[t]*x0
        vlp = y + S[t]*x
        vimp = vbh-vlp
        #  0     1    2     3      4     5     6      7            8             9              10
        # "S", "X", "Y", "px", "Vbh", "Vlp", "IL", "Phi", "ArbTrades", "LargeTrades", "SmallTrades"
        df.iat[t, 1] = x
        df.iat[t, 2] = y
        df.iat[t, 3] = exchangeRate() #px
        df.iat[t, 4] = vbh
        df.iat[t, 5] = vlp
        df.iat[t, 6] = vimp
        df.iat[t, 7] = Phi
        df.iat[t, 8] = total_arb_trades
        df.iat[t, 9] = total_large_trades
        df.iat[t, 10] = total_small_trades
        
        S_ = S[t]
        x_ = x
        y_ = y
    
    #Vbh = y0 + S[t] * x0
    #Vlp = y + S[t] * x
    #ImpL = Vbh - Vt
    
    return df

def roundDF(df, decimals, columns):
    for c in columns:
        df[c] = df[c].astype('float').round(decimals)

def SStats(S):
    dS = np.diff(np.log(S))
    print("diff dS")
    print(stats.describe(dS))


if __name__=="__main__":
    alpha = 0.5
    
    beta = alpha/(1-alpha)
    y0 = 1000_000
    
    np.random.seed(42)
    N= 200_000
    #N = 400
    [S, ts] = load_data(N=N)
    SStats(S)
    print(stats.describe(S))
    tau_vec = [0.0001, 0.0005, 0.0010, 0.0015, 0.0030]
    prob_trade_vec = [1,1,0]#[0, 0.25, 0.4, 0.5]
    prob_small_vec = [1,0,0]#[0, 0.5, 1]

    dfSummary = pd.DataFrame()
    k = 0
    for prob_trade in prob_trade_vec:
        p = prob_small_vec[k]
        for tau in tau_vec:
            df = sim(p, prob_trade, y0, tau)
            roundDF(df, 2, ["Y", "Vbh", "Vlp", "IL", "Phi"])
            dfInfo = pd.DataFrame.from_dict({"tau (bps)":[tau*1e4], "ProbLTTrade": [prob_trade], "PSmallTrade": [p]})
            dfLastSim = df.iloc[[df.shape[0]-1]].copy()
            dfLastSim.drop(["px", "S"], axis=1, inplace=True)
            dfSummary = pd.concat([dfSummary, \
                pd.concat([dfInfo, dfLastSim.reset_index(drop=True)], axis=1)],
                axis=0)
            T = S.shape[0] - 1
            E = prob_trade * ((1-p)*2+p)
        display(dfSummary)    
        k+=1

    roundDF(dfSummary, 2, ["Y", "Vbh", "Vlp", "IL", "Phi"])
    display(dfSummary)
    num = len(glob.glob(f"results/*{N}*.csv"))
    dfSummary.to_csv(f"./results/result{N}_{num}.csv")
    
    #fig, ax = plt.subplots()
    #plt.bar(prob_small_vec, impLoss[1,], width=0.1)
    #plt.show()
    # plot
    #fig, ax = plt.subplots()
    #ax.plot(ts, S, 'b-', linewidth=2.0, label='S*')
    #ax.plot(ts, px, 'r-', linewidth=0.5, label='AMM Exchange Rate')
    #ax.legend()
    #plt.show()