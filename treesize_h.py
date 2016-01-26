import numpy as np
import random
import funcEval
from tree_subt import add_subt_cf
from scipy.optimize.minpack import curve_fit_2
from tree2func import tree2f
from eval_str import eval_
from g_address import get_address
from speciation import ind_specie


def eval_prob(population):
    n_nodes=[]
    for ind in population:
        n_nodes.append(len(ind))
    nn_nodes=np.asarray(n_nodes)
    av_size=np.mean(nn_nodes)
    c=1.5
    for ind in population:
        ind.LS_probability(0.)
        y=c-(len(ind)/av_size)
        if y>1.:
            ps=1
            ind.LS_probability(1.)
        elif (y>=0. and y<=1.):
            ps=y
            ind.LS_probability(y)

def trees_h(population, n):
    eval_prob(population)
    for ind in population:
        if random.random()<ind.get_LS_prob():
            strg=ind.__str__() #convierte en str el individuo
            l_strg=add_subt_cf(strg) #le anade el arbol y lo convierte en arreglo
            c = tree2f() #crea una instancia de tree2f
            cd=c.convert(l_strg) #convierte a l_strg en infijo
            xdata,ydata=get_address(n)
            #outp=open('ls_ind.txt', 'a')
            #outp.write('\n%s;%s;%s' %(ind.get_params(),ind, cd))
            beta_opt, beta_cov, info, msg, success= curve_fit_2(eval_,cd , xdata, ydata, p0=ind.get_params() ,full_output=1, maxfev=400)
            if success not in [1, 2, 3, 4]:
                ind.LS_applied_set(0)
            else:
                ind.LS_applied_set(1)
            ind.params_set(beta_opt)
            funcEval.cont_evalp=funcEval.cont_evalp+info['nfev']


#tomar las especies y aplicarles la heuristica
def specie_h(population,n):
    for ind in population:
       if ind.bestspecie_get()==1:
            strg=ind.__str__() #convierte en str el individuo
            l_strg=add_subt_cf(strg) #le anade el arbol y lo convierte en arreglo
            c = tree2f() #crea una instancia de tree2f
            cd=c.convert(l_strg) #convierte a l_strg en infijo
            xdata,ydata=get_address(n)

            beta_opt, beta_cov, info, msg, success= curve_fit_2(eval_,cd , xdata, ydata, p0=ind.get_params() ,full_output=1, maxfev=400)
            if success not in [1, 2, 3, 4]:
                ind.LS_applied_set(0)
            else:
                ind.LS_applied_set(1)
            ind.params_set(beta_opt)
            funcEval.cont_evalp=funcEval.cont_evalp+info['nfev']

#como determinar los mejores de cada especie
def best_specie(population,n):
    eval_prob(population)
    for ind in population:
        if ind.bestspecie_get()==1:
            if random.random()<ind.get_LS_prob():
                strg=ind.__str__() #convierte en str el individuo
                l_strg=add_subt_cf(strg) #le anade el arbol y lo convierte en arreglo
                c = tree2f() #crea una instancia de tree2f
                cd=c.convert(l_strg) #convierte a l_strg en infijo
                xdata,ydata=get_address(n)

                beta_opt, beta_cov, info, msg, success= curve_fit_2(eval_,cd , xdata, ydata, p0=ind.get_params() ,full_output=1, maxfev=250000)
                if success not in [1, 2, 3, 4]:
                    ind.LS_applied_set(0)
                else:
                    ind.LS_applied_set(1)
                ind.params_set(beta_opt)
                funcEval.cont_evalp=funcEval.cont_evalp+info['nfev']

