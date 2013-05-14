def raman_molecular(z,wave_0,wave_r):
    """
    Function for generating molecular extinction coefficients at original and
    Raman shifted wavelengths.  Ozone absorption is ignored.

    Inputs:

    z = altitude [m]
    wave_0 = original wavelength [nm]
    wave_r = Raman-shifted wavelength [nm]

    Outputs:

    alpha_0 = extinction coefficient at original [1/m]
    alpha_r = extinction at Raman
    """
    import numpy as np
    import lidar_tools_v1 as ltools
    
    beta = np.empty_like(z)
    alpha = np.empty_like(z)

    #calculate temperature and pressure profiles using US Standard atmosphere
    
    [T,P,d] = ltools.std_atm(z)

    #determine backscatter and extinction coefficients as per Rayleigh scattering
    #equations and constats from Kovalev pp 33-36

    T_s = 288.15  #[K] reference temperature
    P_s = 101325.0  #[Pa] reference pressure
    N_s = 2.547e25  #[1/m^3]  reference number concentration
    gamma = 0.0279 #[unitless] depolarization factor (from Kovalev pg.35)

    #calculate reference index of refraction using a polynomial approximation

    nu_0 = 1000/wave_0 #frequency in 1/um
    m_s0 = 1 + 1e-8*(8342.13+(2406030/(130-nu_0**2))+(15997/(38.9-nu_0**2)))

    #now calculate index of refraction at altitude as function of temperature and pressure

    m_0 = 1+(m_s0-1)*((1+0.00367*T_s)/(1+0.00367*T))*(P/P_s)

    #convert air mass density to number density

    N_a = 6.02214e23 #Avogadro's number [#/mol]
    M_air = 0.028964 #molar density of air [kg/mol]

    N = N_a*d/M_air


    #without absorption, extinction is equal to total scattering

    alpha_0 = (8*np.pi**3*(m_0**2-1)**2*N/(3*N_s**2*(wave_0*1e-9)**4))*((6+3*gamma)/(6-7*gamma))* \
            (P/P_s)*(T_s/T)


    #repeat above steps for Raman shifted wavelength

    nu_r = 1000/wave_r #frequency in 1/um
    m_sr = 1 + 1e-8*(8342.13+(2406030/(130-nu_r**2))+(15997/(38.9-nu_r**2)))

    #now calculate index of refraction at altitude as function of temperature and pressure

    m_r = 1+(m_sr-1)*((1+0.00367*T_s)/(1+0.00367*T))*(P/P_s)

    #without absorption, extinction is equal to total scattering

    alpha_r = (8*np.pi**3*(m_r**2-1)**2*N/(3*N_s**2*(wave_r*1e-9)**4))*((6+3*gamma)/(6-7*gamma))* \
            (P/P_s)*(T_s/T)

    return T,P,d,alpha_0,alpha_r


def nitrocon(z):
    """
    Function for calculating the number concentration of
    Nitrogen at a given altitude

    Inputs:

    z = altitude [m]

    Outputs:

    n_N2 = number concentration of Nitrogen [#/m^3]
    
    """
    import numpy as np
    import lidar_tools_v1 as ltools
    
    #calculate temperature and pressure profiles using US Standard atmosphere

    [T,P,d] = ltools.std_atm(z)

    #determine backscatter and extinction coefficients as per Rayleigh scattering
    #equations and constats from Kovalev pp 33-36

    T_s = 288.15  #[K] reference temperature
    P_s = 101325.0  #[Pa] reference pressure
    N_s = 2.547e25  #[1/m^3]  reference number concentration
    gamma = 0.0279 #[unitless] depolarization factor (from Kovalev pg.35)

    #convert air mass density to number density

    N_a = 6.02214e23 #Avogadro's number [#/mol]
    M_air = 0.028964 #molar density of air [kg/mol]

    n_t = N_a*d/M_air


    #assuming ideal gas, volume fraction of Nitrogen is mole fraction

    c_N2 = 0.78084

    n_N2 = n_t*c_N2

    return n_N2

def raman_molprof(z,wave_0, wave_r, T_0 = (1.0,1.0)):
    """
    Function for generating a theoretical profile of normalized attenuated
    N2 vibrational Raman backscatter.  In other words, this provides
    an array that can be multiplied by lidar output power and system constant
    to provide a Raman response profile

    Inputs:

    z = an array of altitudes [m]
    wave_0 = lidar wavelength [nm]
    wave_r = Raman shifted wavelength [nm]
    T_0 = tuple describing transmissivity to the first z value [unitless],
         at each wavelength, defaults to (1.0,1.0) (perfect transmition)

    Outputs:
    P_out = a class object with the following attributes
    z = altitude
    vals = vector of normalized attenuated backscatter [unitless]
    beta = vector of backscatter coefficients [1/m*sr]
    alpha = vector of extinction coefficients [1/m]
    tau = total optical depth of the column
    
    """
    import numpy as np
    import lidar_tools_v1 as ltools

    T = np.empty_like(z)
    P = np.empty_like(z)
    d = np.empty_like(z)
    n_N2 = np.empty_like(z)
    alpha_0 = np.empty_like(z)
    alpha_r = np.empty_like(z)
    P_z = np.empty_like(z)

    for n in range(len(z)):
        [T[n],P[n],d[n],alpha_0[n],alpha_r[n]] = raman_molecular(z[n], wave_0, wave_r)
        n_N2[n] = nitrocon(z[n])
        
    T_total = T_0[0]*T_0[1]
    
    P_z[0] = z[0]**-2*n_N2[0]*T_total
    
    for n in range(1,len(z)):
        T_step = np.exp(-(alpha_0[n] + alpha_r[n])*(z[n]-z[n-1]))
        T_total = T_total*T_step
        P_z[n] = z[n]**-2*n_N2[n]*T_total

    class Raman:
        pass
    P_out = Raman()
    P_out.vals = P_z
    P_out.z = z
    P_out.T = T
    P_out.P = P
    P_out.d = d
    P_out.n_N2 = n_N2
    P_out.alpha_m0 = alpha_0
    P_out.alpha_mr = alpha_r
    P_out.alpha_p0 = np.zeros_like(alpha_0)
    P_out.alpha_pr = np.zeros_like(alpha_r)
    P_out.alpha_t0 = P_out.alpha_m0 + P_out.alpha_p0
    P_out.alpha_tr = P_out.alpha_mr + P_out.alpha_pr
    P_out.wave_0 = wave_0
    P_out.wave_r = wave_r
    P_out.tau = T_total

    return P_out


def addlayer(P_in, layer, u = 1.0):
    """
    Function that adds a layer of known backscatter coefficient
    and lidar ratio onto an existing lidar response profile

    Inputs:

    P_in = a class object containing the input profile with key/value pairs
    as defined in molprof

    layer = a dict object containing the following
    z = altitude values within the layer [m]
    alpha = array of extinction coefficents at these altitudes [1/m*sr]
    
    u = angstrom coefficient of the added layer [unitless] defaults to 1.0

    Outputs:
    P_out = dict object like P_in with layer added

    """
    import numpy as np
    from copy import deepcopy
    
    P_out = deepcopy(P_in)
    
    alpha_in = layer['alpha']
    z_in = layer['z']

    z_min = min(z_in)
    z_max = max(z_in)
    
    for n in range(len(P_out.z)):
        if P_out.z[n] < z_min:
            T_total = (P_out.vals[n]/P_out.vals[0])*(P_out.n_N2[0]/P_out.n_N2[n])*(P_out.z[n]/P_out.z[0])**2
        elif P_out.z[n] <= z_max:
            P_out.alpha_p0[n] = P_out.alpha_p0[n] + np.interp(P_out.z[n],z_in,alpha_in)
            P_out.alpha_pr[n] = P_out.alpha_p0[n] * (P_out.wave_0/P_out.wave_r)**u
            P_out.alpha_t0[n] = P_out.alpha_m0[n] + P_out.alpha_p0[n]
            P_out.alpha_tr[n] = P_out.alpha_mr[n] + P_out.alpha_pr[n]
            T_step = np.exp(-(P_out.alpha_t0[n] + P_out.alpha_tr[n])*(P_out.z[n]-P_out.z[n-1]))
            T_total = T_total*T_step
            P_out.vals[n] = P_out.z[n]**-2*P_out.n_N2[n]*T_total
        else:
            T_step = np.exp(-(P_out.alpha_t0[n] + P_out.alpha_tr[n])*(P_out.z[n]-P_out.z[n-1]))
            T_total = T_total*T_step
            P_out.vals[n] = P_out.z[n]**-2*P_out.n_N2[n]*T_total

    return P_out

           
if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt

    z = np.arange(100,15000,3,dtype=np.float)

    wave_0 = 532.0
    wave_r = 607.0
    

    R_mol = raman_molprof(z,wave_0,wave_r)

    z_layer = np.arange(2000,6000,5,dtype=np.float)
                     
    alpha_layer = np.ones_like(z_layer)*5e-5

    layer = {'z':z_layer,'alpha':alpha_layer}

    R_1 = addlayer(R_mol,layer,1.0)

    rangecor = np.empty_like(R_1.vals)

    for n in range(len(z)):
        rangecor[n] = R_1.vals[n]*(z[n]**2)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,3,1)
    ax1.plot(R_mol.T,z)
    ax1.set_xlabel('Temperature [K]')
    ax1.set_ylabel('Height [m]')

    ax2 = fig1.add_subplot(1,3,2)
    ax2.plot(R_mol.P,z)
    ax2.set_xlabel('Pressure [Pa]')

    ax3 = fig1.add_subplot(1,3,3)
    ax3.plot(R_mol.n_N2,z)
    ax3.set_xlabel('Nitrogen Number Density [#/m^3]')

    fig2 = plt.figure()
    ax1 = fig2.add_subplot(1,3,1)
    ax1.plot(rangecor/rangecor[0],z)
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
  
    
