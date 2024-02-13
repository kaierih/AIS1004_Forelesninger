from numpy import sin, cos, pi, exp
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed, FloatSlider, IntSlider, HBox, VBox, interactive_output, Layout


def getArrow(x, y, dx, dy, arrowhead_scale=1):
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy,dx)
    len_arrowhead = min(arrowhead_scale/16, r/2)
    x_arrow = np.array([x, x+dx, x+dx+len_arrowhead*cos(theta-4*pi/5), x+dx, x+dx+len_arrowhead*cos(theta+4*pi/5)])
    y_arrow = np.array([y, y+dy, y+dy+len_arrowhead*sin(theta-4*pi/5), y+dy, y+dy+len_arrowhead*sin(theta+4*pi/5)])
    return x_arrow, y_arrow

def sliderPanelSetup(set_details, n_of_sets=1, slider_type='float'):
    panel_col = []
    sliders = {}
    for i in range(n_of_sets):
        panel_row = []
        for item in set_details:
            mathtext = item['description']
            mathtext = mathtext.strip('$')
            if n_of_sets > 1:
                if mathtext.find(" ") == -1:
                    mathtext = '$' + mathtext + '_' + str(i+1) + '$' 
                else:
                    mathtext = '$' + mathtext.replace(" ", '_'+str(i+1)+'\ ', 1) + '$'
            else:
                mathtext = '$' + mathtext + '$'
            #mathtext = r'{}'.format(mathtext)

            panel_row.append(FloatSlider(value=item['value'], 
                                         min=item['min'],
                                         max = item['max'], 
                                         step = item['step'], 
                                         description=mathtext, 
                                         layout=Layout(width='95%')))
            
            sliders[item['keyword']+str(i+1)] = panel_row[-1]
        panel_col.append(HBox(panel_row, layout = Layout(width='100%')))
    layout = VBox(panel_col, layout = Layout(width='90%'))
    return sliders, layout

class vectorPlot:
    def __init__(self, ax, A_max, N=1):
        self.ax = ax
        self.N = N
        init_values = np.zeros((2, N))
        self.lines = self.ax.plot(init_values, init_values)
        self.ax.grid(True)
        self.ax.set_xlabel("Reell akse")
        self.ax.set_ylabel("Imaginær akse")
        self.ax.axis([-A_max, A_max, -A_max, A_max])
        
    def update(self, x_new_lines, y_new_lines):
        assert len(x_new_lines)==len(y_new_lines)==self.N, 'Error: mismatch between x and y dimensions.'
        for i in range(self.N):
            x_line = x_new_lines[i]
            y_line = y_new_lines[i]
            L = len(x_line)
            assert len(y_line)==L, 'Error: mismatch between x and y dimensions.'
            x_arrows = np.zeros((L-1)*5)
            y_arrows = np.zeros((L-1)*5)
            for j in range(1, L):
                b = j*5
                a = b-5
                x_arrows[a:b], y_arrows[a:b] = getArrow(x_line[j-1], y_line[j-1], x_line[j]-x_line[j-1], y_line[j]-y_line[j-1])
            self.lines[i].set_xdata(x_arrows)
            self.lines[i].set_ydata(y_arrows)
            
    def setLabels(self, names):
        self.ax.legend(self.lines, names, loc='upper right')
        
    def setStyles(self, styles):
        for i in range(min(len(styles), len(self.lines))):
            try:
                self.lines[i].set_color(styles[i]['color'])
            except:
                pass
            
            try:
                self.lines[i].set_linestyle(styles[i]['linestyle'])
            except:
                pass 
        

class timeSeriesPlot:
    def __init__(self, ax, t, A_max, N=1, t_unit='s'):
        res  = len(t)
        self.N = N
        t_nd = np.outer(t, np.ones(self.N))
        x_t = np.zeros((res, self.N))          

        self.ax = ax
        self.lines = self.ax.plot(t_nd, x_t)
        
        # avgrensning av akser, rutenett, merkede punkt på aksene, tittel, aksenavn
        self.ax.axis([t[0], t[-1], -A_max, A_max])
        self.ax.grid(True)
        self.ax.set_xticks(np.linspace(t[0],t[-1],11))
        self.ax.set_xlabel("Tid (" + t_unit + ")")
        
    def update(self, new_lines):
        assert self.N == len(new_lines), "Error: Parameter lenght different from number of sines."
        for i in range(self.N):
            self.lines[i].set_ydata(new_lines[i])
            
    def setLabels(self, names):
        self.ax.legend(self.lines, names, loc='upper right')
        
    def setStyles(self, styles):
        for i in range(min(len(styles), len(self.lines))):
            try:
                self.lines[i].set_color(styles[i]['color'])
            except:
                pass
            
            try:
                self.lines[i].set_linestyle(styles[i]['linestyle'])
            except:
                pass
# Demo 1
# Visualisering av en sinusbølge
class SineWaveDemo():
    def __init__(self, fig_num=1, fig_size = (9, 4)):
        # Set up canvas
        plt.close(fig_num)
        self.fig = plt.figure(fig_num, figsize=fig_size)
        
        # Set up subplot with sine wave
        ax = plt.subplot()
        ax.set_title(" ")
        
        self.t = np.linspace(-1, 1, 501)
        self.SineWave = timeSeriesPlot(ax, self.t, A_max = 2)
        
        # Tilpass figur-layout
        self.fig.tight_layout(pad=0.1, w_pad=1.0, h_pad=1.0)
        
        # Set up slider panel
        self.sliders, self.layout = sliderPanelSetup(
            [{'keyword': 'A', 'value': 1, 'min': 0, 'max': 2, 'step': 0.1, 'description': r'A'},
             {'keyword': 'f', 'value': 1, 'min': 0.5, 'max': 15, 'step': 0.5, 'description': r'f'},
             {'keyword': 'phi', 'value': 0.5, 'min': -1, 'max': 1, 'step': 1/12, 'description': r'\phi (\times \pi)'}])
        
        # Run demo
        out = interactive_output(self.update, self.sliders)
        display(self.layout, out)
        
    def update(self, **kwargs):
        x1 = kwargs['A1']*cos(2*pi*self.t*kwargs['f1'] + kwargs['phi1']*pi)
        titleStr = '$x(t)='+str(kwargs['A1'])+"\cdot\cos(2\pi\cdot"+str(kwargs['f1'])+"\cdot t +"+str(round(kwargs['phi1'],2))+"\pi)$" # Plot-tittel
        titleStr = titleStr.replace("+-", "-")
        self.SineWave.ax.set_title(titleStr)
        self.SineWave.update([x1])


# Demo 2:
# Fourierrekke-dekomposisjon av firkantpuls
class SquareDecompDemo():
    def __init__(self, fig_num=1, fig_size=(9,4)):
        # Set up canvas
        plt.close(fig_num)
        self.fig = plt.figure(fig_num, figsize=fig_size)
        
        # Set up subplot with sine wave
        ax = plt.subplot()
        ax.plot([0, 0.5, 0.5, 1], [1, 1, -1, -1], 'r-.') # Trace Square Wave
        ax.set_title(" ")
        
        self.t = np.linspace(0, 1, 501)
        self.SquareWave = timeSeriesPlot(ax, self.t, A_max = 1.3)
        
        # Tilpass figur-layout
        self.fig.tight_layout(pad=0.1, w_pad=1.0, h_pad=1.0)
        
        # Set up slider panel
        self.sliders, self.layout = sliderPanelSetup(
            [{'keyword': 'N', 'value': 1, 'min': 1, 'max': 31, 'step': 2, 'description': r'N'}])
        
        # Run demo
        out = interactive_output(self.update, self.sliders)
        display(self.layout, out)
        
    def update(self, **kwargs):
        T = self.t[-1]
        x_t = 0*self.t
        for k in range(1,round(kwargs['N1'])+1, 2):
            x_t+= 4/(k*pi)*sin(2*pi*self.t*k/T)
        self.SquareWave.update([x_t])
        self.SquareWave.ax.set_title('Tilnærming til firkantpuls med de '+str(round(kwargs['N1']))+' første fourierrekke-koeffisientene.')

        


        
# Demo 3: 
# Vektorrepresentasjon av komplekst tall på polarform        
class ComplexPolarDemo():
    def __init__(self, fig_num=1, fig_size=(7,7)):
        # Set up canvas
        plt.close(fig_num)
        self.fig = plt.figure(fig_num, figsize=fig_size)

        # Set up vector subplot
        ax = plt.subplot()
        x_circ = cos(np.linspace(0, 2*pi, 101))
        y_circ = sin(np.linspace(0, 2*pi, 101))    
        ax.plot(x_circ,y_circ, 'r:')
        
        ax.set_title(r"Vektorrepresentasjon av komplekst tall $r\cdot e^{j\phi}$")
        
        self.VectorFig = vectorPlot(ax, A_max = 1.3, N = 3)
        
        self.VectorFig.setStyles([{'color': 'tab:green', 'linestyle': '-.'},
                                      {'color': 'tab:orange', 'linestyle': '-.'},
                                      {'color': 'tab:blue'}])
        
        self.VectorFig.setLabels([r'Reell del',
                                  r'Imaginær del'])
        
        # Tilpass figur-layout
        self.fig.tight_layout(pad=0.1, w_pad=1.0, h_pad=1.0)

        # Set up slider panel
        self.sliders, self.layout = sliderPanelSetup(
            [{'keyword': 'r', 'value': 1, 'min': 0.3, 'max': 1.3, 'step': 0.1, 'description': r'r'},
             {'keyword': 'phi', 'value': 0.25, 'min': -1, 'max': 1, 'step': 1/12, 'description': r'\phi\ (\times\ \pi)'}]) 
        
        # Run Demo:
        out = interactive_output(self.update, self.sliders)
        display(self.layout, out)
        
    def update(self, **kwargs):
        x = kwargs['r1']*cos(kwargs['phi1']*pi)
        y = kwargs['r1']*sin(kwargs['phi1']*pi)
        self.VectorFig.update([[0, x], 
                               [x, x], 
                               [0, x]],
                              [[0, 0],
                               [0, y],
                               [0, y]])
        
# Demo 4:
# Sum av sinusbølger med vektoraddisjon        
class VectorSumDemo():
    def __init__(self, fig_num=1, fig_size=(9, 4)):
        # Set up canvas
        plt.close(fig_num)
        self.fig = plt.figure(fig_num, figsize=fig_size)
        
        
        # Set up subplot with sine waves
        ax1 = plt.subplot(1, 5, (1,3))
        ax1.set_title(r"Sum av sinusbølger med frekvens $f=1Hz$")
        
        self.t = np.linspace(0, 2, 501)
        self.SineWaves = timeSeriesPlot(ax1, self.t, A_max = 2,  N = 3)
        
        self.SineWaves.setStyles([{'color': 'tab:green', 'linestyle': '-.'},
                                  {'color': 'tab:orange', 'linestyle': '-.'},
                                  {'color': 'tab:blue'}])
        
        self.SineWaves.setLabels([r'$x_1(t) = A_1\cdot \cos(2\pi t + \phi_1)$',
                                  r'$x_2(t) = A_2\cdot \cos(2\pi t + \phi_2)$', 
                                  r'$y(t)=x_1(t)+x_2(t)$'])

        
        # Set up vector subplot
        ax2 = plt.subplot(1, 5, (4,5))
        ax2.set_title("Kompleks amplitude $a_k = A_ke^{j\phi_k}$")
        ax2.set_aspect(1)
        
        self.VectorSumPlot = vectorPlot(ax2, A_max = 2, N = 3)
        
        self.VectorSumPlot.setStyles([{'color': 'tab:green', 'linestyle': '-.'},
                                      {'color': 'tab:orange', 'linestyle': '-.'},
                                      {'color': 'tab:blue'}])
        
        # Adjust figure layout
        self.fig.tight_layout(pad=0.1, w_pad=1.0, h_pad=1.0)

        # Set up slider panel
        self.sliders, self.layout = sliderPanelSetup(
            [{'keyword': 'A', 'value': 1, 'min': 0, 'max': 1, 'step': 0.1, 'description': r'A'},
             {'keyword': 'phi', 'value': 0.5, 'min': -1, 'max': 1, 'step': 1/12, 'description': r'\phi (\times \pi)'}],
            n_of_sets = 2)
        
        # Run demo
        out = interactive_output(self.update, self.sliders)
        display(self.layout, out)
        
    def update(self, **kwargs):
        x1 = kwargs['A1']*cos(2*pi*self.t + kwargs['phi1']*pi)
        x2 = kwargs['A2']*cos(2*pi*self.t + kwargs['phi2']*pi)
        y = x1 + x2
        
        self.SineWaves.update([x1, x2, y])
        
        v1_x = np.array([0, kwargs['A1']*cos(kwargs['phi1']*pi)])
        v1_y = np.array([0, kwargs['A1']*sin(kwargs['phi1']*pi)])
        
        v2_x = np.array([0, kwargs['A2']*cos(kwargs['phi2']*pi)])+v1_x[-1]
        v2_y = np.array([0, kwargs['A2']*sin(kwargs['phi2']*pi)])+v1_y[-1]
        
        v3_x = np.array([0, v2_x[-1]])
        v3_y = np.array([0, v2_y[-1]])
        
        self.VectorSumPlot.update([v1_x, v2_x, v3_x], [v1_y, v2_y, v3_y])
              
        
# Demo 5: 
# Vektorrepresentasjon av komplekst tall på polarform        
class demo5:
    def __init__(self, A, omega, T, fig_num=5):
        self.A = np.array(A)         # Rotating vector amplitudes
        self.omega = np.array(omega) # Rotating frequencies
        self.T = T                   # Max time range
        self.L = len(A)              # Number of rotating vectors
        
        # Set up canvas
        plt.close(fig_num)
        self.fig = plt.figure(fig_num, figsize=(7, 7))
        
        ax = plt.subplot()
    
        # Set up vector subplot
        t = np.linspace(0, self.T, 501)
        z_t = np.array([sum(self.A*exp(1j*self.omega*l)) for l in t])
        x_trace = np.real(z_t)
        y_trace = np.imag(z_t)
        ax.plot(x_trace,y_trace, 'r:')
      
        ax.set_title(r"Sum av komplekse eksponentialer.")
        
        self.VectorFig = vectorPlot(ax, A_max = sum(np.absolute(A)), N = 1)
        
        self.VectorFig.setStyles([{'color': 'tab:blue'}])
        
      
        # Tilpass figur-layout
        self.fig.tight_layout(pad=0.1, w_pad=1.0, h_pad=1.0)

        # Set up slider panel
        self.sliders, self.layout = sliderPanelSetup(
            [{'keyword': 't', 'value': 0, 'min': 0.0, 'max': T, 'step': T/100, 'description': r"Tid\ \ \ t"},]) 
        
        # Run Demo:
        out = interactive_output(self.update, self.sliders)
        display(self.layout, out)
        
    def update(self, **kwargs):
        t = kwargs['t1']
        vectors = self.A*np.exp(1j*self.omega*t)
        vectorSums = np.array([np.sum(vectors[0:i]) for i in range(self.L+1)])
        x = np.append(np.array([0]), np.real(vectorSums))
        y = np.append(np.array([0]), np.imag(vectorSums))
        self.VectorFig.update([x], [y])

# Demo 6
# Visualisering av frekvensmiksing
class FrequencyMixingDemo():
    def __init__(self, fig_num=1, fig_size = (9, 4)):
        # Set up canvas
        plt.close(fig_num)
        self.fig = plt.figure(fig_num, figsize=fig_size)
        
        # Set up subplot with sine wave 1
        ax1 = plt.subplot(2,2,1)
        ax1.set_title(" ")
        
        self.t_x = np.linspace(0, 1, 201)
        self.SineWave1 = timeSeriesPlot(ax1, self.t_x, A_max = 1.2)
        
        # Set up subplot with sine wave 2 
        ax2 = plt.subplot(2,2,2)
        ax2.set_title(" ")
        
        self.SineWave2 = timeSeriesPlot(ax2, self.t_x, A_max = 1.2)
        
        # Set up subplot with product 
        ax3 = plt.subplot(2,2,(3,4))
        ax3.set_title(r"$y(t) = x_1(t)\cdot x_2(t)$")
        
        self.t_y = np.linspace(0, 2, 401)
        self.MixedWaves = timeSeriesPlot(ax3, self.t_y, A_max = 1.2)
        
        # Tilpass figur-layout
        self.fig.tight_layout(pad=0.1, w_pad=1.0, h_pad=1.0)
        
        # Set up slider panel
        self.sliders, self.layout = sliderPanelSetup(
            [{'keyword': 'f', 'value': 1, 'min': 0.5, 'max': 15, 'step': 0.5, 'description': r'f'},
             {'keyword': 'phi', 'value': 0.5, 'min': -1, 'max': 1, 'step': 1/12, 'description': r'\phi (\times \pi)'}],
            n_of_sets=2)
        
        # Run demo
        out = interactive_output(self.update, self.sliders)
        display(self.layout, out)
        
    def update(self, **kwargs):
        x1 = cos(2*pi*self.t_y*kwargs['f1'] + kwargs['phi1']*pi)
        x2 = cos(2*pi*self.t_y*kwargs['f2'] + kwargs['phi2']*pi)
        
        y = x1*x2
        
        titleStr1 = "$x_1(t)=\cos(2\pi\cdot"+str(kwargs['f1'])+"\cdot t +"+str(round(kwargs['phi1'],2))+"\pi)$" # Plot-tittel
        titleStr1 = titleStr1.replace("+-", "-")
        self.SineWave1.ax.set_title(titleStr1)
        self.SineWave1.update([x1[0:201]])
        
        titleStr2 = "$x_2(t)=\cos(2\pi\cdot"+str(kwargs['f2'])+"\cdot t +"+str(round(kwargs['phi2'],2))+"\pi)$" # Plot-tittel
        titleStr2 = titleStr2.replace("+-", "-")
        self.SineWave2.ax.set_title(titleStr2)
        self.SineWave2.update([x2[0:201]])
        
        self.MixedWaves.update([y])