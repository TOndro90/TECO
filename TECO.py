##########################################
#  Temperature evolution T_0 of the IGM  #
##########################################

################################################################
# This file includes all the python functions and constants 
# to calculate the temperature evolution T_0.
# If you find this code useful in your research, please consider 
# citing following work:
# Ondro, Arya & Gális (arXiv:2412.11909)
################################################################
import numpy as np
from numpy import loadtxt
Mpc_cm = 3.08568025e24 # cm
Mpc_km = Mpc_cm * 1.0e-5 # km
H100_s = 100. / Mpc_km # s^-1

def Cooling_rate_Collisional_excitationHI( temp, n_e, n_HI ):
    temp_5 = temp/1e5
    return 7.5e-19*np.exp( -118348/temp )*n_e*n_HI/( 1+temp_5**0.5 )

def Cooling_rate_Collisional_excitationHeII( temp, n_e, n_HeII ):
    temp_5 = temp/1e5
    return 5.54e-17*temp**( -0.397 )*np.exp( -473638/temp )*n_e*n_HeII/( 1+temp_5**0.5 )

def Cooling_rate_Collisional_ionizationHI( temp, n_e, n_HI ):
    temp_5 = temp/1e5
    return 1.27e-21*temp**( 0.5 )*np.exp( -157809.1/temp )*n_e*n_HI/( 1+temp_5**0.5 )

def Cooling_rate_Collisional_ionizationHeI( temp, n_e, n_HeI ):
    temp_5 = temp/1e5
    return 9.38e-22*temp**( 0.5 )*np.exp( -285335.4/temp )*n_e*n_HeI/( 1+temp_5**0.5 )

def Cooling_rate_Collisional_ionizationHeII( temp, n_e, n_HeII ):
    temp_5 = temp/1e5
    return 4.95e-22*temp**( 0.5 )*np.exp( -631515/temp )*n_e*n_HeII/( 1+temp_5**0.5 )

def Cooling_rate_RecombinationHII( temp, n_e, n_HII ):
    temp_3 = temp/1e3
    temp_6 = temp/1e6
    return 8.7e-27*temp**( 0.5 )*temp_3**( -0.2 )*n_e*n_HII/( 1+temp_6**0.7 )

def Cooling_rate_RecombinationHeII( temp, n_e, n_HeII ):
    return 1.55e-26*temp**( 0.3647 )*n_e*n_HeII

def Cooling_rate_RecombinationHeIII( temp, n_e, n_HeIII ):
    temp_3 = temp/1e3
    temp_6 = temp/1e6
    return 3.48e-26*temp**( 0.5 )*temp_3**( -0.2 )*n_e*n_HeIII/( 1+temp_6**0.7 )

def Cooling_rate_Dielectronic_recombinationHeII( temp, n_e, n_HeII ):
    return 1.24e-13*temp**( -1.5 )*np.exp( -470000/temp )*( 1+0.3*np.exp(-94000/temp ))*n_e*n_HeII

def Cooling_rate_Free_Free( temp, n_HII, n_HeII, n_HeIII, n_e ):
    gff = 1.1+0.34*np.exp( -(( 5.5-np.log10( temp ))**2 )/3 )
    return 1.42e-27*gff*temp**( 0.5 )*( n_HII+n_HeII+4*n_HeIII )*n_e

def Compton_cooling( temp, z, n_e ):
    return 5.41e-36*n_e*temp*( 1+z )**4

def Recombination_rateHII( temp ):
    temp_3 = temp/1e3
    temp_6 = temp/1e6
    return 8.4e-11*temp**( -0.5 )*temp_3**( -0.2 )/( 1+temp_6**0.7 )

def Recombination_rateHeII( temp ):
    return 1.5e-10*temp**( -0.6353 )

def Recombination_rate_d( temp ):
    return 1.9e-3*temp**( -1.5 )*np.exp( -470000/temp )*( 1+0.3*np.exp( -94000/temp ))

def Recombination_rateHeIII( temp ):
    temp_3 = temp/1e3
    temp_6 = temp/1e6
    return 3.36e-10*temp**( -0.5 )*temp_3**( -0.2 )/( 1+temp_6**0.7 )

def Collisional_ionization_rate_eHI( temp ):
    temp_5 = temp/1e5
    return 5.85e-11*temp**( 0.5 )*np.exp( -157809.1/temp )/( 1+temp_5**0.5 )

def Collisional_ionization_rate_eHeI( temp ):
    temp_5 = temp/1e5
    return 2.38e-11*temp**( 0.5 )*np.exp( -285335.4/temp )/( 1+temp_5**0.5 )

def Collisional_ionization_rate_eHeII( temp ):
    temp_5 = temp/1e5
    return 5.68e-12*temp**( 0.5 )*np.exp( -631515/temp )/( 1+temp_5**0.5 )

def TECO_STEP( T0, n_HI, n_HII, n_HeI, n_HeII, n_HeIII, n_e, dt, Treecool, z, a, H ):
    M_p = 1.6726219e-24
    k_b = 1.380649e-16
    ##########
    T0_init = T0
    n_HI_init = n_HI
    n_HII_init = n_HII
    n_HeI_init = n_HeI
    n_HeII_init = n_HeII
    n_HeIII_init = n_HeIII
    n_e_init = n_e
    n_tot = n_HI+n_HII+n_HeI+n_HeII+n_HeIII+n_e
    ##########
    Photoionization_HI = np.interp( z, 10**Treecool[:,0]-1, Treecool[:,1] )
    Photoionization_HeI = np.interp( z, 10**Treecool[:,0]-1, Treecool[:,2] )
    Photoionization_HeII = np.interp( z, 10**Treecool[:,0]-1, Treecool[:,3] )
    Photoheating_HI = np.interp( z, 10**Treecool[:,0]-1, Treecool[:,4] )
    Photoheating_HeI = np.interp( z, 10**Treecool[:,0]-1, Treecool[:,5] )
    Photoheating_HeII = np.interp( z, 10**Treecool[:,0]-1, Treecool[:,6] )
    ##########
    H_tot = Photoheating_HI*n_HI+Photoheating_HeI*n_HeI+Photoheating_HeII*n_HeII
    RC_tot = Cooling_rate_RecombinationHII(T0, n_e, n_HII)+Cooling_rate_RecombinationHeII(T0, n_e, n_HeII)+Cooling_rate_RecombinationHeIII(T0, n_e, n_HeIII)+Cooling_rate_Dielectronic_recombinationHeII(T0, n_e, n_HeII)
    CC_tot = Cooling_rate_Collisional_excitationHI(T0, n_e, n_HI)+Cooling_rate_Collisional_excitationHeII(T0, n_e, n_HeII)+Cooling_rate_Collisional_ionizationHI(T0, n_e, n_HI)+Cooling_rate_Collisional_ionizationHeI(T0, n_e, n_HeI)+Cooling_rate_Collisional_ionizationHeII(T0, n_e, n_HeII)
    FF = Cooling_rate_Free_Free(T0, n_HII, n_HeII, n_HeIII, n_e)
    Compton = Compton_cooling(T0, z, n_e)
    dQdt = H_tot-RC_tot-CC_tot-FF-Compton
    ##########
    dens = ( n_HI+n_HII+4*( n_HeI+n_HeII+n_HeIII ) )*M_p
    X_HI = M_p*n_HI/dens
    X_HII = M_p*n_HII/dens
    X_HeI = M_p*n_HeI/dens
    X_HeII = M_p*n_HeII/dens
    X_HeIII = M_p*n_HeIII/dens
    X_e = M_p*n_e/dens
    X_tot = X_HI+X_HII+X_HeI+X_HeII+X_HeIII+X_e
    ##########
    C_HI = Recombination_rateHII(T0)*n_HII*n_e
    D_HI = Photoionization_HI+Collisional_ionization_rate_eHI(T0)*n_e
    n_HI = ( C_HI*dt+n_HI )/( 1+D_HI*dt )
    ##########
    C_HII = Photoionization_HI*n_HI+Collisional_ionization_rate_eHI(T0)*n_HI*n_e
    D_HII = Recombination_rateHII(T0)*n_e
    n_HII = ( C_HII*dt+n_HII )/( 1+D_HII*dt )
    ##########
    C_HeI = Recombination_rateHeII(T0)*n_HeII*n_e+Recombination_rate_d(T0)*n_HeII*n_e
    D_HeI = Photoionization_HeI+Collisional_ionization_rate_eHeI(T0)*n_e
    n_HeI = ( C_HeI*dt+n_HeI )/( 1+D_HeI*dt )
    ##########           
    C_HeII = Photoionization_HeI*n_HeI+Collisional_ionization_rate_eHeI(T0)*n_HeI*n_e+Recombination_rateHeIII(T0)*n_HeIII*n_e
    D_HeII = Photoionization_HeII+Collisional_ionization_rate_eHeII(T0)*n_e+Recombination_rateHeII(T0)*n_e+Recombination_rate_d(T0)*n_e
    n_HeII = ( C_HeII*dt+n_HeII )/( 1+D_HeII*dt )
    ########## 
    C_HeIII = Photoionization_HeII*n_HeII+Collisional_ionization_rate_eHeII(T0)*n_HeII*n_e
    D_HeIII = Recombination_rateHeIII(T0)*n_e
    n_HeIII = ( C_HeIII*dt+n_HeIII )/( 1+D_HeIII*dt )
    ########## 
    C_e = Photoionization_HI*n_HI+Photoionization_HeI*n_HeI+Photoionization_HeII*n_HeII+Collisional_ionization_rate_eHI(T0)*n_HI*n_e+Collisional_ionization_rate_eHeI(T0)*n_HeI*n_e+Collisional_ionization_rate_eHeII(T0)*n_HeII*n_e
    D_e = Recombination_rateHII(T0)*n_HII+Recombination_rateHeII(T0)*n_HeII+Recombination_rate_d(T0)*n_HeII+Recombination_rateHeIII(T0)*n_HeIII
    n_e = ( C_e*dt+n_e )/( 1+D_e*dt )
    ########## 
    dXdt = M_p*(C_e-D_e*n_e)/dens
    T0 = T0_init-2*H*T0_init*dt+( 2*dQdt/( 3*k_b*n_tot ))*dt - ( T0_init*dXdt/X_tot )*dt
    n_HI = n_HI-3*H*n_HI_init*dt
    n_HII = n_HII-3*H*n_HII_init*dt
    n_HeI = n_HeI-3*H*n_HeI_init*dt
    n_HeII = n_HeII-3*H*n_HeII_init*dt
    n_HeIII = n_HeIII-3*H*n_HeIII_init*dt
    n_e = n_e-3*H*n_e_init*dt   
    return T0, n_HI, n_HII, n_HeI, n_HeII, n_HeIII, n_e

################################################################
z_start = 8
z_end = 1
n_calc = 10000
T_start = 100
Treecool = loadtxt(r"fg20_def.dat")
Omega_L = 1-0.308
Omega_M = 0.308
Omega_K =0
Omega_b = 0.0482
h = 0.678
H0 = h * H100_s
Y = 0.24            
################################################################
G = 6.6743e-8
M_p = 1.6726219e-24
##########
z_values = np.linspace(z_start, z_end, n_calc)
a_values = 1/( z_values+1 )
T0 = np.zeros( len( z_values ) )
n_HI = np.zeros( len( z_values ) )
n_HII = np.zeros( len( z_values ) )
n_HeI = np.zeros( len( z_values ) )
n_HeII = np.zeros( len( z_values ) )
n_HeIII = np.zeros( len( z_values ) )
n_e = np.zeros( len( z_values ) )
rho_c = ( 3*H0**2 )/( 8*np.pi*G )
rho_gas_mean = Omega_b*rho_c
n_HI[0] = ( 1-Y )*rho_gas_mean/( M_p*a_values[0]**3 )
n_HeI[0] = Y*rho_gas_mean/(4*M_p*a_values[0]**3)
T0[0] = T_start
##########
for i in range( 1,len(z_values) ):
    da = a_values[i] - a_values[i-1]
    H = H0*np.sqrt( Omega_M/a_values[i]**3 + Omega_L )
    a_dot = H0*np.sqrt( Omega_M/a_values[i] + Omega_L*a_values[i]**2 )
    dt = da/a_dot
    T0[i], n_HI[i], n_HII[i], n_HeI[i], n_HeII[i], n_HeIII[i], n_e[i] = TECO_STEP(T0[i-1], n_HI[i-1], n_HII[i-1], n_HeI[i-1], n_HeII[i-1], n_HeIII[i-1], n_e[i-1], dt, Treecool, z_values[i], a_values[i], H)
    

from matplotlib import pyplot as plt
plt.plot(z_values, T0)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    