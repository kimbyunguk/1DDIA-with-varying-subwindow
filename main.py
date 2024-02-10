# import necessary modules
import os, sys
import numpy             as np               
import matplotlib.pyplot as plt
from scipy.io            import loadmat, savemat

# Import pre-defined functions
from wavnumest import WNE

Cond = "B"                                           # Bathymetry condition
Case = np.char.add(Cond, ['1', '2', '3', '4', '5'])  # Case indices
Case_num = len(Case)

fdir = 'Plae the location of input data'
sdir = 'Plae the location where to save the result'

x    =  loadmat(fdir+'res_{}'.format(Case[0]))['x'][:,:,None][0,:,0]
t    =  loadmat(fdir+'res_{}'.format(Case[0]))['t'][:,:,None][0,:,0]
lx, lt = len(x), len(t)
dx, dt = x[1]-x[0], t[1]-t[0]

valid_t, valid_x = np.asarray([0,-1]), (np.asarray([0,2000])/dx).astype('int')

for c in Case:
    eta  =  loadmat(fdir+'res_{}'.format(c))['eta'][:,:,None]
    U    =  loadmat(fdir+'res_{}'.format(c))['U'][:,:,None]
    d    =  loadmat(fdir+'res_{}'.format(c))['depth'][0,:,None]
    if c == Case[0]:
        eta_stack = eta
        U_stack   = U
        d_stack   = d
    else:
        eta_stack = np.concatenate((eta_stack,eta),axis=2)
        U_stack   = np.concatenate((U_stack,U),axis=2)
        d_stack   = np.concatenate((d_stack,d),axis=1)
        
eta_stack      =  eta_stack[valid_t[0]:valid_t[1]:,valid_x[0]:valid_x[1],:]
eta_norm_stack =  np.zeros_like(eta_stack)
d_stack        =  d_stack[valid_x[0]:valid_x[1],:]
t, x           =  t[valid_t[0]:valid_t[1]], x[valid_x[0]:valid_x[1]]

for n in range(eta_stack.shape[1]):
    for m in range(eta_stack.shape[2]):
        locals()['data'] = eta_stack[:,n,m]
        normfac = np.nanmax(data)- np.nanmin(data)
        data    = (data-np.nanmin(data))/(normfac) 
        eta_norm_stack[:,n,m] = data

""" Pysical Parameters """

g      =  9.81
T0     =  8
omega  =  2*np.pi/T0
k0     =  (omega**2)/g

""" 
Wavenumber Estimation
"""

jump         = 20

RMSE, Dep_est = [], []
Dep_est.append(x)

windowsize_search = np.arange(80, 1000+40, 40)
wincase_num = len(windowsize_search)

for case in [1,2,3,4,5]:
    Dep_est.append(d_stack[:,case-1])
    
    for i, windowsize in enumerate(windowsize_search):

        print('Case = {} | Windowsize = {}'.format(Case[case-1], windowsize))    
                
        interr_point, wavenumber = WNE.PhsOpt(eta_norm_stack[:,:,case-1],
                                              windowsize,
                                              jump,
                                              x,
                                              t,
                                              Opt = 'phGOSH'
                                              )

        intmaxlen = len(interr_point)
        
        if i == 0:
            globals()['Dep_Case{}'.format(case)] = np.ones((intmaxlen,wincase_num*2), dtype=float) * np.nan
            globals()['WN_Case{}'.format(case)] = np.ones((intmaxlen,wincase_num*2), dtype=float) * np.nan            
        
        L0  = (2*np.pi)/k0
        wavelength = 2*np.pi/wavenumber
        d = (1/wavenumber) * np.arctanh(omega**2/g/wavenumber)
        
        RMSE.append(np.sqrt(np.nanmean((d+d_stack[0,case-1])**2)))
        
        """
        Visualize the results
        """
        
        plt.plot(interr_point,-d,'o',color='r', label='Estimation')
        plt.plot(x,d_stack[:,case-1],'-',color='k', label='GroundTruth')
        plt.ylim(np.min(d_stack[:,case-1])*1.1,0)
        plt.xlabel('x [m]', fontsize = 12)
        plt.ylabel('Bottom elevation [m]', fontsize = 12)
        plt.text(x[0],d_stack[0,case-1]*0.3,
                  "Case = {}"
                  "\n"
                  "Window size = {:.0f} m"
                  "\n".format(Case[case-1],int(windowsize*dx)))
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        plt.savefig(sdir+'DepthInversionResult_{}{}_{:.0f}m.png'.format(Cond,case,windowsize*dx))
        plt.pause(1)
        plt.close()
        
        globals()['Dep_Case{}'.format(case)][:intmaxlen,2*i]    =  interr_point
        globals()['Dep_Case{}'.format(case)][:intmaxlen,2*i+1]  =  -d

        globals()['WN_Case{}'.format(case)][:intmaxlen,2*i]    =  interr_point
        globals()['WN_Case{}'.format(case)][:intmaxlen,2*i+1]  =  wavenumber         


    """
    Save the results in mat format
    """
    savemat(sdir + '/Res_{}{}.mat'.format(Cond,case),
            {"Dep_est" : globals()['Dep_Case{}'.format(case)],
              "WN_est" : globals()['WN_Case{}'.format(case)],
              "windowsize_cases" : windowsize_search,
              "eta_stack":  eta_stack,
              "d_stack" :  d_stack,
              "x" :  x,
              "t" :  t,
              })

os._exit(00)
sys.modules[__name__].__dict__.clear()


