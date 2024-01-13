from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from edNEGmodel import *
from solve_edNEGmodel import solve_edNEGmodel

I_stim = 36e-12  # [A]
alpha = 2
t_dur = 30       # [s]
stim_start = 10
stim_end = 20

sol, my_cell = solve_edNEGmodel(t_dur, alpha, I_stim, stim_start, stim_end)
t = sol.t

phi_sn, phi_se, phi_sg, phi_dn, phi_de, phi_dg, phi_msn, phi_mdn, phi_msg, phi_mdg = my_cell.membrane_potentials()
E_Na_sn, E_Na_sg, E_Na_dn, E_Na_dg, E_K_sn, E_K_sg, E_K_dn, E_K_dg, E_Cl_sn, E_Cl_sg, E_Cl_dn, E_Cl_dg, E_Ca_sn, E_Ca_dn = my_cell.reversal_potentials()

plt.figure(1)
plt.plot(t, phi_msn*1000, '-', label='soma')
plt.plot(t, phi_mdn*1000, '-', label='dendrite')
plt.title('Neuronal membrane potentials')
plt.xlabel('time [s]')
plt.legend(loc='upper right')

plt.figure(100)
plt.plot(t, phi_msg*1000, '-', label='somatic layar')
plt.plot(t, phi_mdg*1000, '-', label='dendritic layer')
plt.title('Glial membrane potentials')
plt.xlabel('time [s]')
plt.legend(loc='upper right')

plt.show()

