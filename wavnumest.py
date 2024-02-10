import numpy as np               
from pyGOSH import GlobalOptimization as go

class WNE:
    def PhsOpt(data, windowsize, jump, x, t, Opt):
        
        dx = x[1]-x[0]

        xdata = np.arange(0,windowsize,1)*dx
        
        len_T, len_x = data.shape
        
        interr_point    =  np.arange(windowsize//2, len_x-windowsize//2, jump)
        interr_point_x  =  interr_point*dx + x[0]
        
        k = []  
        for i, p in enumerate(interr_point):           
            ran = [p-windowsize//2, p+windowsize//2]
            eta_window = data[:, ran[0]:ran[1]]
            
            if np.isnan(eta_window).any():
                continue
            
            eta_fft = np.fft.fft(eta_window, axis=0)[:len_T//2,:].T
            C = np.dot(eta_fft, eta_fft.conj().T) / eta_fft.shape[1]
            eigenvalues, eigenvectors = np.linalg.eigh(C)

            idx = np.argsort(eigenvalues)[::-1]
            eigvecs = eigenvectors[:,idx][:,0]
            phasemap = np.unwrap(np.angle(eigvecs))
            
            f, fft = np.fft.fftfreq(windowsize, dx)[:windowsize//2], np.fft.fft(phasemap)[:windowsize//2]
            psd = abs(fft)
            ind = np.argmax(psd)
            
            if ind==1:
                x1_range = np.arange(1/f[ind+1],windowsize*2,0.2)
            else:
                x1_range = np.arange(1/f[ind+1],1/f[ind-1],0.2)
                                     
            x2_range = np.arange(0,2*np.pi,np.pi/90)

            if Opt == 'phGOSH':
            
                def testobjsin(x):
                    obj2 = 2*np.pi/x[0]*xdata+x[1]
                    f = np.sum((phasemap-obj2)**2)
                    return f
                     
                optimizer = go.Optimizer(testobjsin,
                        lb = [x1_range[0], x2_range[0]],
                        ub = [x1_range[-1], x2_range[-1]],
                        algorithm = 'MCCE',
                        stop_step = 250,
                        stop_span = 1e-4,
                        stop_obj_percent = 0.1, # %
                        stop_fcal = 1e7,
                        dimension_restore = True,
                        n_complex = 4,
                        n_complex_size = 8,
                        iseed = None,
                        iniflg = None,
                        pop_init_method = 'LHS',
                        init_pop = None,
                        verbose = True,
                        algorithm_params = 'default',
                        n_jobs = 1,
                        pre_dispatch = '2*n_jobs',
                        obj_eval = 'serial',
                        int_program = [False,False]
                        ) 
    
                for i in range(2):
                    optimizer.evolve()
                 
                bestparams = optimizer.bestXs[-1]
                
            else:
                print('Error: Optimization scheme is not valid')
                
            k.append(2*np.pi/bestparams[0])
            
        return interr_point_x, np.asarray(k)
                            