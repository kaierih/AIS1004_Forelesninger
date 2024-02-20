from numpy import sin, cos, pi, exp, real, imag
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from ipywidgets import interact, fixed, FloatSlider, IntSlider, HBox, VBox, interactive_output, Layout
import ipywidgets as widget
from numpy.typing import ArrayLike

class FreqRespDemo:
    def __init__(self, b, a, fig_num=1, figsize=(8,6)):
        plt.close(fig_num)
        self.fig = plt.figure(fig_num, figsize=figsize)
        
        self.res=150
        w_c = np.max(np.abs(np.roots(a)))
        f_c = w_c/(2*np.pi)
        self.x_c = int(np.log10(f_c))
        self.f = np.logspace(self.x_c-1, self.x_c+2, self.res+1)
        
        self.b = b
        self.a = a
        self.M = len(b)
        self.N = len(a)
        _, Hw = sig.freqs(b, a, worN=2*np.pi*self.f)
        self.Hw_amp = np.abs(Hw)
        self.Hw_phase = np.unwrap(np.angle(Hw, deg=True), period=360.0)
        
        
        # Amplituderespons
        ax11 = plt.subplot(2,2,1)
        ax11.semilogx(self.f, self.Hw_amp)
        ax11.set_xlim([self.f[0], self.f[-1]])
        ax11.set_ylim(ymin=0)
        #ax11.set_xticks(np.linspace(0, 1, 5)*pi)
        #ax11.set_xticklabels([r'$'+str(round(i,2))+'\pi$' for i in np.linspace(0, 1, 5)])
        ax11.set_xlabel(r'Frekvens $f$ (Hz)')
        ax11.set_ylabel(r'Forsterking (Gain) ')
        ax11.grid(True)
        ax11.set_title('placeholder')
        self.ax11 = ax11
        
        # Markør for valgt frekvens:
        self.ampMarker, = ax11.plot([self.f[0]], [self.Hw_amp[0]], 'oC3')
        

        # Faserespons
        ax12 = plt.subplot(2,2,2)
        ax12.semilogx(self.f, self.Hw_phase)
        phaseLabels = ax12.get_yticks()
        phaseLim = ax12.get_ylim()
        ax12.set_yticks(phaseLabels)
        ax12.set_ylim(phaseLim)
        #ax12.set_yticklabels([r'$'+str(round(i,2))+'\pi$' for i in phaseLabels])
        ax12.set_xlim([self.f[0], self.f[-1]])
        #ax12.set_xticks(np.linspace(0, 1, 5)*pi)
        #ax12.set_xticklabels([r'$'+str(round(i,2))+'\pi$' for i in np.linspace(0, 1, 5)])
        ax12.set_xlabel(r'Frekvens $f$ (Hz)')
        ax12.set_ylabel(r'Fase $\theta$ (grader)')
        ax12.grid(True)
        ax12.set_title('placeholder')
        self.ax12 = ax12
        
        # Markør for valgt frekvens:
        self.phaseMarker, = plt.plot([self.f[0]], [self.Hw_phase[0]], 'oC3')

        # Tidsvindu for signal
        self.t = np.linspace(0, 1/self.f[0], 2*self.res+1)
        
        # Sinusfigurer
        ax2 = plt.subplot(2,2,(3,4))
        ax2.set_title('placeholder')
        self.V_inn, = plt.plot(self.t, np.sin(2*np.pi*self.f[0]*self.t), color='tab:blue', label=r'$v_{inn}(t)$')
        self.V_ut, = plt.plot(self.t, self.Hw_amp[0]*np.sin(2*np.pi*self.f[0]*self.t + self.Hw_phase[0]), color='tab:red', label=r'$v_{ut}(t)$')
        ax2.legend(loc="upper right")
        ax2.set_xlim([self.t[0], self.t[-1]])
        ax2.grid(True)
        ax2.set_xlabel("Tid $t$ (sekund)")
        ax2.set_ylabel("Spenning (Volt)")
        self.ax2 = ax2
        
       
        # Confiugre Layout
        self.fig.tight_layout()
        
        #Set up slider panel
        freq_slider = widget.FloatSlider(
                                    value = self.x_c-1,
                                    min=self.x_c-1,
                                    max=self.x_c+2-3/self.res,
                                    step = 3/self.res,
                                    description=r' $\text{Log Frekvens: }f = 10^{\wedge}$',
                                    disabled=False,
                                    style = {'description_width': 'initial'},
                                    layout=Layout(width='95%'),
                                    continuous_update=True
                                    )
        self.layout = VBox([freq_slider])
        self.userInput = {'f': freq_slider}
        
        # Run demo
        out = interactive_output(self.update, self.userInput)
        display(self.layout, out)

    
    def update(self, f):
        T = 10**-np.floor(f)
        index = int((f-self.x_c+1)/3*self.res)

        self.ampMarker.set_xdata([self.f[index]])
        self.ampMarker.set_ydata([self.Hw_amp[index]])
        self.phaseMarker.set_xdata([self.f[index]])
        self.phaseMarker.set_ydata([self.Hw_phase[index]])
        
        self.ax11.set_title(r"$\left| H \left(j2\pi \cdot %.2f \right) \right| = %.2f$"%(self.f[index], self.Hw_amp[index]))
        self.ax12.set_title(r"$\angle H \left(j2\pi \cdot %.2f \right) = %.1f^{\circ}$"%(self.f[index], self.Hw_phase[index]))
        titlestr = r"""$v_{inn}(t) = \sin(2\pi\cdot %.2f \cdot t), \ \ \ \ \ v_{ut}(t) = %.2f\cdot\sin(2\pi\cdot %.2f \cdot t +%.1f^{\circ}), \ \ \ \ \ 0 \leq t < %.3f$
                    """%(self.f[index],self.Hw_amp[index], self.f[index], self.Hw_phase[index], T) 
        titlestr=titlestr.replace("+-", "-")
        self.ax2.set_title(titlestr)        
        
        xt = sin(2*np.pi*self.f[index]*self.t)
        yt = self.Hw_amp[index]*sin(2*np.pi*self.f[index]*self.t+np.deg2rad(self.Hw_phase[index]))

        if(T != self.ax2.get_xlim()[1]):
            self.t = np.linspace(0, T, 2*self.res+1)
            self.V_inn.set_xdata(self.t)
            self.V_ut.set_xdata(self.t)
            self.ax2.set_xlim([0, T])
        self.V_inn.set_ydata(xt)
        self.V_ut.set_ydata(yt)
        self.fig.tight_layout()