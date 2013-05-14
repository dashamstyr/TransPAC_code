def addnoise(P_in,background = 0,stdev = []):
    #Adds gaussian random noise and background signal to any profile
    #if no standard deviation is defined, the noise added is standard
    #shot noise - poisson distrobution approximated by a gaussian with
    #std = sqrt(signal)
    
    import numpy as np
    import random
    from copy import deepcopy
    
    P_out = deepcopy(P_in)
    
    if stdev:
        for n in range(len(P_out.vals)):
            P_out.vals[n] = P_out.vals[n] + random.gauss(background,stdev)
            #print random.gauss(background,stdev)
    else:
        for n in range(len(P_out.vals)):
            stdev = np.sqrt(P_out.vals[n]+background)
            P_out.vals[n] = P_out.vals[n] + random.gauss(background,stdev)

    return P_out

def background_subtract(P_in):
    #takes advantage of inverse square law to calculate background signal for
    #a profile

    #start by selecting a regionof sufficient altitude that the r quared law
    #will make the ratio of signal to background sufficiently low

    import numpy as np
    from copy import deepcopy

    P_out = deepcopy(P_in)
    
    z_min = 6000 #[m]

    #select data from altitudes higher than z_min and muliply full signal by
    #range squared, if this gives less than 500 values, take uppermost 500

    z = P_out.z[np.where(P_out.z >= z_min)]

    if len(z) <= 500:
        z = P_out.z[-500:]

    r_sq = P_out.vals[-len(z):]*z**2

    #since background is constant while signal is reduced as z^2, first
    #coefficient is equal to background
    
    coeffs = np.polyfit(z,r_sq,2,full=False)
    

    background = coeffs[0]

    P_out.backsub_vals = P_out.vals-background
    P_out.back = background

    return P_out

def numdiff_d1(m_in,z):
    #generate a matrix that is the numerical differentiation of input vector
    #or w.r.t first dimension of a 2-D matrix

    import numpy as np
    from scipy import sparse

    dims = np.shape(m_in)

    data1 = np.ones(dims[0])
    data2 = -1*data1
    

    diags = [1,0]

    diffmat = sparse.spdiags([data1,data2],diags,dims[0],dims[0])

    d_m = diffmat.dot(m_in)
    
    d_z = diffmat.dot(z)
    
    dm_dz = d_m/d_z
    
    
    #set up buffer assuming dm_dz is constant for laswt two values
    
    try:
        dm_dz[-1,:] = dm_dz[-2,:]
    except IndexError:
        dm_dz[-1] = dm_dz[-2]

    return dm_dz

def numdiff_d2(m_in,x):
    #generate a matrix that is the numerical differentiation of input vector
    #or w.r.t first dimension of a 2-D matrix

    import numpy as np
    from scipy import sparse

    dims = np.shape(m_in)

    data1 = np.ones(dims[1])
    data2 = -1*data1
    

    diags = [-1,0]

    diffmat = sparse.spdiags([data1,data2],diags,dims[1],dims[1])
    
    #print diffmat

    d_m = m_in.dot(diffmat.todense())
    
    #print d_m
    
    d_x = x.dot(diffmat.todense())
    
    print d_x
    
    dm_dx = d_m/d_x
    
    #set up buffer assuming dm_dz is constant 
    
    dm_dx[:,-1] = dm_dx[:,-2]

    return dm_dx

def raman_analytical(P_in,u = 1.0):
    #Uses analytical inversion of Raman scattering equation to calculate values
    #for particulate extinction from background-subtracted signal

    import numpy as np

    #fisrt perform background subtraction on the profile if it hasn't already happened

    if P_in.backsub_vals:
        from copy import deepcopy
        P_out = deepcopy(P_in)
    else:
        P_out = background_subtract(P_in)
    
    #now calculate an intermediate value x to be differentiated
    
    x = np.log(P_out.n_N2/(P_out.vals*P_out.z**2))

    x_diff = numdiff_d1(x,P_out.z)

    #estimate particulate extinction coefficient using equation from Kovalev eq 11.6 from pg.394

    alpha_p0 = (x_diff - P_out.alpha_m0 - P_out.alpha_mr)/(1+(P_out.wave_0/P_out.wave_r)**u)

    P_out.alpha_p0 = alpha_p0

    #use this to determine the rest of the extinction terms

    P_out.alpha_pr = alpha_p0*(P_out.wave_0/P_out.wave_r)**u

    P_out.alpha_t0 = P_out.alpha_p0 + P_out.alpha_m0
    P_out.alpha_tr = P_out.alpha_pr + P_out.alpha_mr

    #calculate estimated signal based on this value of alpha_p0

    T_total = 1.0
    P_out.backsub_vals[0] = P_out.z**-2*P_out.n_N2[0]*T_total
    for n in range(1,len(P_out.z)):
        T_step = np.exp(-(alpha_t0[n] + alpha_tr[n])*(P_out.z[n]-P_out.z[n-1]))
        T_total = T_total*T_step
        P_out.backsub_vals[n] = P_out.z[n]**-2*P_out.n_N2[n]*T_total

    return P_out
    
def SNR_calc(P_in):
    #estimates signal to noise ratio from an input profile
    #as a function of altitude using a sliding window of 100 pixels assuming for this local section
    #a polynomial curve fit is acceptable to fit the mean value
    
    import numpy as np

    z = P_in.z
    
    try:
        vals = P_in.vals
    except AttributeError:
        print 'Warning: Background subtraction has not been performed prior to SNR calculation'
        vals = P_in.vals
    
    winsize = 20
    
    stdev = np.empty_like(vals)
    
    for n in range(len(vals)-winsize):
        z_win = z[n:(n+winsize)]
        v_win = vals[n:(n+winsize)]
    
        coeffs = np.polyfit(z_win,v_win,2,full=False)
        
        baseline = coeffs[0]*z_win**2 + coeffs[1]*z_win + coeffs[2]
        
        noise = v_win-baseline
        
        stdev[n] = np.std(noise)
        
    stdev[n:] = stdev[n]
    
    SNR = vals/stdev
    
    return SNR


#def avgnoise(signum,SNR_out):
    
if __name__== "__main__":

    import raman_tools_v1 as rtools
    import numpy as np
    import matplotlib.pyplot as plt
    from copy import deepcopy

    bkg = 0 #mV
    const = 1e-20  #combines pulse energy, Raman scattering x-sec, and optical
                    #path efficiency
    alpha_p = 5e-5 #extinction coefficient for added layer
    

    z = np.arange(100,15000,3,dtype=np.float)

    wave_0 = 532.0
    wave_r = 607.0
    

    R_mol = rtools.raman_molprof(z,wave_0,wave_r)

    z_layer = np.arange(2000,6000,5,dtype=np.float)
                     
    alpha_layer = np.ones_like(z_layer)*alpha_p

    layer = {'z':z_layer,'alpha':alpha_layer}

    R_1 = rtools.addlayer(R_mol,layer,1.0)

    P_mol = deepcopy(R_mol)
    P_mol.vals = P_mol.vals*const

    P_1 = deepcopy(R_1)
    P_1.vals = P_1.vals*const

    P_noisy = addnoise(P_1,background = bkg, stdev = 1e-4)

    P_noisy = background_subtract(P_noisy)

    rangecor_0 = np.empty_like(P_mol.vals)
    rangecor_1 = np.empty_like(P_1.vals)
    rangecor_n = np.empty_like(P_noisy.vals)

    SNR = SNR_calc(P_noisy)

    for n in range(len(z)):
        rangecor_0[n] = P_mol.vals[n]*(z[n]**2)
        rangecor_1[n] = P_1.vals[n]*(z[n]**2)
        rangecor_n[n] = P_noisy.vals[n]*(z[n]**2)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,3,1)
    ax1.plot(SNR,z)
    ax1.set_xlabel('SNR')
    ax1.set_ylabel('Height [m]')

    ax2 = fig1.add_subplot(1,3,2)
    ax2.plot(P_1.vals,z)
    ax2.set_xlabel('Lidar Return [counts]')

    ax3 = fig1.add_subplot(1,3,3)
    ax3.plot(R_mol.T,z)
    ax3.set_xlabel('Temperature [K]')

    fig2 = plt.figure()
    ax1 = fig2.add_subplot(1,3,1)
    ax1.plot(rangecor_0,z,rangecor_1,z,rangecor_n,z)
    ax1.set_xscale('log')
    ax1.set_xlabel('Range corrected Raman signal multiplier')
    ax1.set_ylabel('Height [m]')

    ax2 = fig2.add_subplot(1,3,2)
    ax2.plot((R_1.alpha_t0),z)
    ax2.set_xlabel('Extinction coefficient - laser wavelength [1/m]')

    ax3 = fig2.add_subplot(1,3,3)
    ax3.plot((R_1.alpha_tr),z)
    ax3.set_xlabel('Extinction coefficient - Raman wavelength [1/m]')

    plt.show()
    
   
    
