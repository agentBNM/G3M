import pandas as pd

def roundToMil(v):
    return round(v/1e6,3)

if __name__=="__main__":

    df = pd.read_csv("./results/result200000_2.csv")
    dfSummary = df.loc[:,["tau (bps)", "ProbLTTrade", "PSmallTrade", "Vlp", "IL", "Phi", "ArbTrades", "LargeTrades", "SmallTrades"]]
    numTrades = dfSummary["ArbTrades"]+dfSummary["LargeTrades"]+dfSummary["SmallTrades"]
    
    dfSummary["IL/#T"] = dfSummary["IL"]/numTrades
    dfSummary["IL/#T"] = dfSummary["IL/#T"].astype('float').round(1)
    dfSummary["RelErr"] = (dfSummary["Phi"]-dfSummary["IL"])/(dfSummary["Vlp"]+dfSummary["IL"])*100
    dfSummary["RelErr"] = dfSummary["RelErr"].astype('float').round(1)
    dfSummary.columns = ["$\tau$", "$p$", "$p^{(S)}$", "$V_T$", "${\Psi}_T$", "$\hat{\Psi}_T$", "\#Arb", "\#Large", "\#Small", "IL/\#T", "RelErr"]
    dfSummary["\#Small"]=dfSummary["\#Small"].map('{:,}'.format)
    dfSummary["\#Large"]=dfSummary["\#Large"].map('{:,}'.format)
    dfSummary["\#Arb"]=dfSummary["\#Arb"].map('{:,}'.format)
    dfSummary["$V_T$"] = roundToMil(dfSummary["$V_T$"])
    dfSummary["${\Psi}_T$"] = roundToMil(dfSummary["${\Psi}_T$"])
    dfSummary["$\hat{\Psi}_T$"]= roundToMil(dfSummary["$\hat{\Psi}_T$"])
    pd.options.display.float_format = '{:,}'.format
    print(dfSummary.to_latex(index=False, column_format="rrrrrrrrrrr", escape=False, float_format='{:,.3f}'.format))
