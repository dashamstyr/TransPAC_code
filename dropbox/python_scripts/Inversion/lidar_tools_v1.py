def std_atm(z):
    ########################################################################
    # Program that defines temperature, pressure, and density profiles for
    # a US Standard Atmosphere (1976) given a height
    # in meters [AMSL] up to 47,000 m
    # Returns:
    # T = temperature [K]
    # P = pressure [Pa]
    # d = density [kg/m^3]
    # 
    ########################################################################
    
    import numpy as np

    #constants

    g = 9.80665  #gravitational constant [m/s^2]
    M = 0.028964 #molar density of air [kg/mol]
    R = 8.31432  #universal gas constant [N*m/mol/K]

    #Define breakpoints for US Standard atmosphere, with associated altitude,
    #pressure, density, temperature, and dry adiabatic lapse rates

    alt = [0, 11000, 20000, 32000, 47000]
    press = [101325,22632.1, 5474.89,868.019,110.906]
    dense = [1.225, 0.36391, 0.08803, 0.01322, 0.00143]
    temp = [288.15, 216.65, 216.65, 228.65, 270.65]
    lapse = [-0.0065, 0, 0.001, 0.0028, 0]

    #fisrt test to make sure no altitudes exceed the maximum

    if z > max(alt):
        raise VaueError('Sorry, all altitudes must be below %d' %max(alt))

    #start by determining temprature through linear interpolation

    T = np.interp(z,alt,temp)

    #now determine Pressure and density using different functions based on
    #atmospheric region
    
    if alt[0] <= z <= alt[1]:
        P = press[0]*(temp[0]/T)**(g*M/(R*lapse[0]))
        d = dense[0]*(T/temp[0])**((-g*M/(R*lapse[0]))-1)
    elif alt[1] < z <= alt[2]:
        P = press[1]*np.exp(-g*M*(z-alt[1])/(R*temp[1]))
        d = dense[1]*np.exp(-g*M*(z-alt[1])/(R*temp[1]))
    elif alt[2] < z <= alt[3]:
        P = press[2]*(temp[2]/T)**(g*M/(R*lapse[2]))
        d = dense[2]*(T/temp[2])**((-g*M/(R*lapse[2]))-1)
    elif alt[3] < z <= alt[4]:
        P = press[3]*(temp[3]/T)**(g*M/(R*lapse[3]))
        d = dense[3]*(T/temp[3])**((-g*M(R*lapse[3]))-1)

    return T,P,d


def molecular(z,wave):
    """
    Function for generating molecular scattering and extinction coefficients based
    on an altitude and a laser wavelength.  Ozone absorption is ignored.

    Inputs:

    z = altitude [m]
    wave = wavelength [nm]

    Outputs:

    beta = backscatter coefficients [1/m*sr]
    alpha = extinction coefficients [1/m]
    """
    import numpy as np
    
    #note: make sure z is in the form of a numpy matrix
    
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

    nu = 1000/wave #frequency in 1/um
    m_s = 1 + 1e-8*(8342.13+(2406030/(130-nu**2))+(15997/(38.9-nu**2)))

    #now calculate index of refraction at altitude as function of temperature and pressure

    m = 1+(m_s-1)*((1+0.00367*T_s)/(1+0.00367*T))*(P/P_s)

    #convert air mass density to number density

    N_a = 6.02214e23 #Avogadro's number [#/mol]
    M_air = 0.028964 #molar density of air [kg/mol]

    N = N_a*d/M_air


    #without absorption, extinction is equal to total scattering

    alpha = (8*np.pi**3*(m**2-1)**2*N/(3*N_s**2*(wave*1e-9)**4))*((6+3*gamma)/(6-7*gamma))* \
            (P/P_s)*(T_s/T)


    #For Rayleigh scattering the extinction to backscatter ratio is 8*pi/3

    beta = 3*alpha/(8*np.pi)

    return T,P,d,beta,alpha

def molprof(z,wave, T_0 = 1.0):
    """
    Function for generating a theoretical profile of normalized attenuated
    backscatter.  In other words, this provides
    an array that can be multiplied by lidar output power and system constant
    to provide a lidar response profile

    Inputs:

    z = an array of altitudes [m]
    wave = lidar wavelength [nm]
    T_0 = transmissivity to the first z value [unitless], defaults to 1.0 (perfect transmittion)

    Outputs:
    P_out = a class object with the following attributes
    z = altitude
    P_z = vector of normalized attenuated backscatter [unitless]
    beta = vector of backscatter coefficients [1/m*sr]
    alpha = vector of extinction coefficients [1/m]
    tau = total optical depth of the column
    """
    import numpy as np

    T = np.empty_like(z)
    P = np.empty_like(z)
    d = np.empty_like(z)
    beta = np.empty_like(z)
    alpha = np.empty_like(z)
    P_z = np.empty_like(z)

    for n in range(len(z)):
        [T[n],P[n],d[n],beta[n],alpha[n]] = molecular(z[n], wave)
    
    T_total = T_0
    P_z[0] = z[0]**-2*beta[0]*T_total**2
    
    for n in range(1,len(z)):
        T_step = np.exp(-alpha[n]*(z[n]-z[n-1]))
        T_total = T_total*T_step
        P_z[n] = z[n]**-2.0*beta[n]*T_total**2

    class Elastic:
        pass
    P_out = Elastic()
    P_out.vals = P_z
    P_out.z = z
    P_out.T = T
    P_out.P = P
    P_out.d = d
    P_out.beta_m = beta
    P_out.alpha_m = alpha
    P_out.beta_p = np.zeros_like(beta)
    P_out.alpha_p = np.zeros_like(alpha)
    P_out.beta_t = P_out.beta_m+P_out.beta_p
    P_out.alpha_t = P_out.alpha_m+P_out.alpha_p
    P_out.wave = wave
    P_out.tau = T_total

    return P_out


def addlayer(P_in, layer, lrat):
    """
    Function that adds a layer of known backscatter coefficient
    and lidar ratio onto an existing lidar response profile

    Inputs:

    P_in = a class object containing the input profile with key/value pairs
    as defined in molprof

    layer = a dict object containing the following
    z = altitude values within the layer [m]
    beta = array of backscatter coefficents at these altitudes [1/m*sr]
    
    lrat = lidar (extinction to backscater) ratio of the added layer [1/sr]

    Outputs:
    P_out = dict object like P_in with layer added

    """
    import numpy as np
    from copy import deepcopy
    
    P_out = deepcopy(P_in)
    
    beta_in = layer['beta']
    z_in = layer['z']

    beta_layer = []
    z_min = min(z_in)
    z_max = max(z_in)
    
    for n in range(len(P_out.z)):
        if P_out.z[n] < z_min:
            T_total = (P_out.vals[n]/P_out.vals[0])*(P_out.beta_t[0]/P_out.beta_t[n])*(P_out.z[n]/P_out.z[0])**2
        elif P_out.z[n] <= z_max:
            P_out.beta_p[n] = P_out.beta_p[n] + np.interp(P_out.z[n],z_in,beta_in)
            P_out.alpha_p[n] = P_out.alpha_p[n] + P_out.beta_p[n]*lrat
            P_out.beta_t[n] = P_out.beta_p[n] + P_out.beta_m[n]
            P_out.alpha_t[n] = P_out.alpha_p[n] + P_out.alpha_m[n]
            T_step = np.exp(-P_out.alpha_t[n]*(P_out.z[n]-P_out.z[n-1]))
            T_total = T_total*T_step
            P_out.vals[n] = P_out.z[n]**-2*P_out.beta_t[n]*T_total**2
        else:
            T_step = np.exp(-P_out.alpha_t[n]*(P_out.z[n]-P_out.z[n-1]))
            T_total = T_total*T_step
            P_out.vals[n] = P_out.z[n]**-2*P_out.beta_t[n]*T_total**2

    
    P_out.tau = T_total

    return P_out

if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt

    z = np.arange(100,15000,3)
    z = z+0.0

    wave = 532

    P_mol = molprof(z,wave)

    z_layer = np.arange(1000,2000,5,dtype=np.float)

    beta_layer = np.ones_like(z_layer)*5e-6

    layer = {'z':z_layer,'beta':beta_layer}

    P_1 = addlayer(P_mol,layer,30)

    
    p_rangecor0 = P_mol.vals*(z**2)
    p_rangecor1 = P_1.vals*(z**2)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,3,1)
    ax1.plot(P_mol.T,z)
    ax1.set_xlabel('Temperature [K]')
    ax1.set_ylabel('Height [m]')

    ax2 = fig1.add_subplot(1,3,2)
    ax2.plot(P_mol.P,z)
    ax2.set_xlabel('Pressure [Pa]')

    ax3 = fig1.add_subplot(1,3,3)
    ax3.plot(P_mol.d,z)
    ax3.set_xlabel('Density [kg/m^3]')

    fig2 = plt.figure()
    ax1 = fig2.add_subplot(1,3,1)
    ax1.plot(p_rangecor1/p_rangecor1[0],z,p_rangecor0/p_rangecor0[0],z)
    ax1.set_xscale('log')
    ax1.set_xlabel('Range corrected signal multiplier')
    ax1.set_ylabel('Height [m]')

    ax2 = fig2.add_subplot(1,3,2)
    ax2.plot((P_1.beta_m+P_1.beta_p),z)
    ax2.set_xlabel('Total backscatter coefficient [1/m/sr')

    ax3 = fig2.add_subplot(1,3,3)
    ax3.plot((P_1.alpha_m+P_1.alpha_p),z)
    ax3.set_xlabel('Total extinction coefficient [1/m')

    plt.show()
