import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import poisson

class EnergySpectrum:
    def __init__(self, N, omega, delta, gVal) -> None:
        self.N = N
        self.omega = omega
        self.delta = delta
        self.gVal = gVal

    def Model(self, model, g, n_b = None, p = None, Manual = None):
        sz = tensor(qeye(self.N), sigmaz())
        sx = tensor(qeye(self.N), sigmax())
        a = tensor(destroy(self.N), qeye(2))

        sp = tensor(qeye(self.N), sigmap())
        sm = tensor(qeye(self.N), sigmam())

        omega_q = (self.delta + self.omega)
        omega_BS = g**2 / (self.delta + 2*self.omega)

        if model == f'pDSC' or model == f'Deep-Strong':
            if n_b == None or p == None:
                return ValueError(f'n_b and p missing values')
            b = sx*a

            beta = g/self.omega
            D = (2*beta*b.dag() - 2*beta*b).expm()
            E = []

            for N_b in range(n_b):
                ket = tensor(fock(self.N, N_b), qeye(2))

                InnerProduct = (D*ket).overlap(ket)

                E.append(self.omega*N_b - g**2/self.omega - (omega_q/2)*p*(-1)**(N_b)*InnerProduct)

            return np.array(E)

        elif model == f'Rabi' or model == f'rabi':
            H = self.omega*a.dag()*a + (1/2)*(self.delta + self.omega)*sz + g*(a.dag() + a)*sx
        
        elif model == f'BS' or model == f'Bloch-Siegert':
             H = (self.omega + omega_BS*sz)*a.dag()*a + 0.5*(omega_q + omega_BS)*sz - 0.5*omega_BS + g*(a*sp + a.dag()*sm)
            #H = (self.omega - omega_BS*sz)*a.dag()*a + 0.5*(omega_q - omega_BS)*sz + 0.5*omega_BS + g*(a*sp + a.dag()*sm)
        elif model == f'JC' or model == f'Jaynes-Cummings':
            H = self.omega*a.dag()*a + 0.5*(self.delta + self.omega)*sz + g*(a*sp + a.dag()*sm)

        elif model == f'M' or model == f'Manual':
            H = Manual

        else:
            raise ValueError(f'Model variable must be string and spelt correctly. Check documentation for more details on useable variable names.')

        return H

    def Sorted_Lists(self, model, n_b = None, p = None, Model = None):

        sorted_EigenVal = []

        OldEigVal, OldEigVec = EnergySpectrum.Model(self, model, self.gVal[0], n_b, p, Model).eigenstates()

        sorted_EigenVal.append(OldEigVal)

        for g in self.gVal[1:]:

            NewEigVal, NewEigVec = EnergySpectrum.Model(self, model, g, n_b, p, Model).eigenstates()

            sorted_NewEigVal = np.zeros((2*self.N, 1))
            sorted_NewEigVec = []

            for i, old_Vec in enumerate(OldEigVec):
                overlaps = np.array( np.abs([old_Vec.dag() * new_vec for new_vec in NewEigVec]) )

                idx = np.argmax(overlaps)
                sorted_NewEigVal[i][0] = NewEigVal[idx]
                sorted_NewEigVec.append(NewEigVec[idx])
            
            sorted_EigenVal.append(sorted_NewEigVal)

            OldEigVal, OldEigVec = sorted_NewEigVal, sorted_NewEigVec

        Sorted_EigenVal = np.zeros((2*self.N, len(self.gVal)))

        for k in range(2*self.N):
            for l in range(len(self.gVal)):
                Sorted_EigenVal[k][l] = sorted_EigenVal[l][k]

        return Sorted_EigenVal, sorted_NewEigVec
    
    def Plot(self, model, ylim, rescaled = True, n_b = None, p = None, color = '#17becf', linestyles = '-', Model = None):
        
        if model == f'pDSC' or model == f'Deep-Strong':
            plot = []
            for n_b in range(n_b):
                Deep = []
                for g in self.gVal:
                    Deep.append(EnergySpectrum.Model(self, model, g, n_b, p)/self.omega + g**2/(self.omega**2))

                plot.append(plt.plot(self.gVal, Deep, color = color, linestyle='--'))
        
        else:
            
            y = EnergySpectrum.Sorted_Lists(self, model, n_b, p, Model)[0]
            
            if rescaled == False:
                for i in range(self.N):
                    plot = plt.plot(self.gVal, y[i,:]/self.omega, color = color, linestyle = linestyles)
            else:
                for i in range(self.N):
                    plot = plt.plot(self.gVal, y[i,:]/self.omega + self.gVal**2/(self.omega**2), color = color, linestyle = linestyles)
            plt.xlim(self.gVal[0], self.gVal[-1])
            plt.ylim(ylim[0], ylim[-1])

            plt.xlabel(r'$g/\omega_c$')
            plt.ylabel(r'$(E_n + g^2/\omega_c)/\omega_c$')
        
        return plot
    
    def Probability(self, model, g, Poisson = False, Type = f'Bar', n_b = None, p = None, Model = None):
        EigVals, EigVecs = EnergySpectrum.Model(self, model, g, n_b, p, Model).eigenstates()
        overlaps_g = []
        overlaps_e = []
        for i in range(self.N):
            overlap = EigVecs[0].overlap(tensor(fock(self.N, i), basis(2, 0)))
            overlaps_g.append(np.abs(overlap)**2)
            overlap = EigVecs[0].overlap(tensor(fock(self.N, i), basis(2, 1)))
            overlaps_e.append(np.abs(overlap)**2)

        fig, ax = plt.subplots()
        ns = np.arange(self.N)

        if Poisson == True:
            ax.plot(ns, poisson.pmf(ns, g**2), 'k:')
            
        if Type == f'bar' or Type == f'Bar':
            for i in range(len(overlaps_g)):
                ax.add_patch(Rectangle((ns[i]-0.46, 0), 1, np.array(overlaps_g)[i], facecolor=f'g'))
                ax.add_patch(Rectangle((ns[i]-0.46, 0), 1, np.array(overlaps_e)[i]))
        if Type == f'Scatter' or Type == f'scatter':
            ax.plot(ns, np.array(overlaps_g), 'o')
            ax.plot(ns, np.array(overlaps_e), 'x')
        if Type != f'bar' and Type != f'Bar' and Type != f'Scatter' and Type != f'scatter':
            raise ValueError(f'Incorrect Varaible Input')

        plt.ylim(0, max(np.maximum(overlaps_e, overlaps_g))*1.1)
        plt.xlim(0,self.N)
        # plt.xticks(ns)

        plt.xlabel(r'Number of excitations, $n$')
        plt.ylabel(f'Probability')

        # f'Poisson Distribution'
        if Poisson == True:
            plt.legend([f'Poisson Distribution', r'|<e,n|$\Phi_n$>$|^2$', r'|<g,n|$\Phi_n$>$|^2$'])
        if Poisson == False:
            plt.legend([r'|<e,n|$\Phi_n$>$|^2$', r'|<g,n|$\Phi_n$>$|^2$'])