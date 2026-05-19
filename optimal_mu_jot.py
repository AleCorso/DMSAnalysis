import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns

aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y','-']
aa_id = {}
for id, a in enumerate(aa): aa_id[a]= id

WT="NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST"
    
q = 20
L = 178

BEGIN = 18
END = 5

WT_SEQ = np.array([aa_id[char] for char in WT])
WT_SEQ = WT_SEQ[BEGIN:-END]
WT_SEQ

from utils import *

def main():
    print("Let's start")
    exp_data_path = 'exp_data/'
    abs = pd.read_csv(exp_data_path+"ab_classification.csv")
    infile = exp_data_path+'test_risk/data.csv'
    data = pd.read_csv(infile)
    
    outfile = infile

    n_trials = 10
    q = 20 
    L = len(WT_SEQ)

    split_fractions = [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.8,0.85,0.9,0.95,0.975]
    mu_grid = 10**(np.array([-9,-8,-7,-6.5,-6,-5.5,-5,-4.5,-4,-3.5,-3,-2,-1,0]))
    r_values = np.zeros(len(split_fractions))


    def get_cols_to_remove(L, q, WT):
        cols = set()
        for site in range(L):
            base = site * (q+1)
            cols.add(base + q)       # ultima feature del sito
            #cols.add(base + WT[site])     # feature wild-type del sito
        return sorted(cols)

    already_done = data['Ab'].unique()

    for id, row in abs.iterrows():

        AB = row['condition']
        print(AB)

        if AB in already_done: 
            print("Already done")
            continue
        
        try: 
            with open(exp_data_path+"starr_nature/DMS_"+AB+".csv", 'r') as file:
                print("DMS data found")
        except: 
            print("DMS data not found")
            continue         

        with open(exp_data_path+"starr_nature/DMS_"+AB+".csv", 'r') as file:
            line = file.readline().split(sep = ',')
            M = int(line[0])
            N = int(line[1])

            delta = np.zeros((M,N),dtype = np.float32)
            G = np.zeros(M,dtype = np.float64)
            for m in range(M):
                line = file.readline().split(sep = ',')
                delta[m] = [int(line[i]) for i in range(N)]
                G[m] = float(line[-1])
        # remove gap       
        cols_to_remove = get_cols_to_remove(L, q, WT_SEQ)
        cols_to_keep   = [i for i in range(N) if i not in set(cols_to_remove)]
        
        delta_new = delta[:, cols_to_keep]
        delta = delta_new.copy()
        
        risk_values = np.zeros(n_trials)

        for i, test_size in enumerate(split_fractions):

            print(f"{i+1}/{len(split_fractions)}: r = {test_size}")

            for j, mu in enumerate(mu_grid):

                print(f"\t{j+1}/{len(mu_grid)}: hmu = {mu}")

                for trial in range(n_trials):

                    delta_train, delta_test, G_train, G_test = train_test_split(delta, G, test_size=test_size)

                    if j == 0 and trial == 0: r_values[i] = q * L / len(G_train)

                    if mu < 5*10**(-8): g_hat, delta_mean, G_mean  = ridge_regression_stable(delta_train, G_train, mu)
                    else: g_hat, delta_mean, G_mean  = ridge_regression(delta_train, G_train, mu)
                    X_test = delta_test - delta_mean[None,:]
                    G_pred = X_test @ g_hat + G_mean
                    risk_values[trial] = np.mean((G_pred-G_test)**2)

                avg_risk = np.mean(risk_values)
                std_risk = np.std(risk_values)/np.sqrt(n_trials)   
            
                data = pd.concat([data, pd.DataFrame({'Ab': [AB],'q':[q], 'L': [L], 'M': [len(G_train)], 'r': [q * L / len(G_train)], 'mu': [mu], 'R': [avg_risk], 'sR': [std_risk],'nTest': [len(G_test)],'nTrial': [n_trials]})])
    
        data.to_csv(outfile, index = False)    
        data = pd.read_csv(outfile)


if __name__ == "__main__":
    main()
