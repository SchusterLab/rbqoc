"""
analytic.py - Tanay's notebook in a .py file
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
from pylab import *
from qutip import *
from matplotlib import cm
import imageio
from pathos.multiprocessing import ProcessingPool as Pool

def plot_points(b):
    # -X and Y data are switched for plotting purposes
    for k in range(len(b.points)):
        num = len(b.points[k][0])
        dist = [sqrt(b.points[k][0][j] ** 2 +
                     b.points[k][1][j] ** 2 +
                     b.points[k][2][j] ** 2) for j in range(num)]
        if any(abs(dist - dist[0]) / dist[0] > 1e-12):
            # combine arrays so that they can be sorted together
            zipped = list(zip(dist, range(num)))
            zipped.sort()  # sort rates from lowest to highest
            dist, indperm = zip(*zipped)
            indperm = array(indperm)
        else:
            indperm = arange(num)
        if b.point_style[k] == 's':
            b.axes.scatter(
                real(b.points[k][1][indperm]),
                - real(b.points[k][0][indperm]),
                real(b.points[k][2][indperm]),
                s=b.point_size[mod(k, len(b.point_size))],
                alpha=1,
                edgecolor='none',linewidths=0,
                zdir='z',
                color=b.point_color[mod(k, len(b.point_color))],
                marker=b.point_marker[mod(k, len(b.point_marker))])

        elif b.point_style[k] == 'm':
            pnt_colors = array(b.point_color *
                               int(ceil(num / float(len(b.point_color)))))

            pnt_colors = pnt_colors[0:num]
            pnt_colors = list(pnt_colors[indperm])
            marker = b.point_marker[mod(k, len(b.point_marker))]
            s = b.point_size[mod(k, len(b.point_size))]
            b.axes.scatter(real(b.points[k][1][indperm]),
                              -real(b.points[k][0][indperm]),
                              real(b.points[k][2][indperm]),
                              s=s, alpha=1, edgecolor='none',linewidths=0,
                              zdir='z', color=pnt_colors,
                              marker=marker)

        elif b.point_style[k] == 'l':
            color = b.point_color[mod(k, len(b.point_color))]
            b.axes.plot(real(b.points[k][1]),
                           -real(b.points[k][0]),
                           real(b.points[k][2]),
                           alpha=0.75, zdir='z',lw=5,
                           color=color)

def plot_bloch(states):

    b = Bloch()
    b.plot_points=lambda: plot_points(b)
    length = len(states)
    # normalize colors to the length of data ##
    nrm = mpl.colors.Normalize(0,length)
    colors = cm.winter(nrm(range(length))) # options: brg, cool, summer, winter, autumn

    ## add data points from expectation values ##
    b.add_points([expect(sigmax(), states),
                  expect(sigmay(), states),
                  expect(sigmaz(), states)],'m')

    # customize sphere properties ##
    b.point_color = list(colors)
    b.point_marker = ['o'] # options: o, s, d, ^
    b.point_size = [20]
    b.zlpos = [1.1,-1.2]
    b.show()
    
    
def animate_bloch(states, duration=0.1, save_all=False):

    b = Bloch()
    b.vector_color = ['r']
    b.view = [-40,30]
    images=[]
    try:
        length = len(states)
    except:
        length = 1
        states = [states]
    ## normalize colors to the length of data ##
    nrm = mpl.colors.Normalize(0,length)
    colors = cm.cool(nrm(range(length))) # options: cool, summer, winter, autumn etc.

    ## customize sphere properties ##
    b.point_color = list(colors) # options: 'r', 'g', 'b' etc.
    b.point_marker = ['o']
    b.point_size = [30]
    
    for i in range(length):
        b.clear()
        b.add_states(states[i])
        b.add_states(states[:(i+1)],'point')
        if save_all:
            b.save(dirc='tmp') #saving images to tmp directory in current working directory b=Bloch()
            filename="tmp/bloch_%01d.png" % i
        else:
            filename='temp_file.png'
            b.save(filename)
        images.append(imageio.imread(filename))
    imageio.mimsave('bloch_anim.gif', images, duration=duration)

def compute_expectation(state):
    return np.array([expect(sigma, state) for sigma in [sigmax(),sigmay(),sigmaz()]])


def Xpiby2_square(args, show_plot=True):
    # Rx[pi/2]= Ry[pi/2].Rz[pi/2].Ry[-pi/2]

    args['A'] = -1*args['A']
    tlist1, states1 = Ypiby2_square(args, show_plot=False)
    
    args['psi0'] = states1[-1]
    args['x_pulse'] = 0
    args['z_pulse'] = args['Zpi']/2
    tlist2, states2 = evolve_fluxonium_square(args, show_plot=False)
    tlist2 = tlist2 + tlist1[-1]
    
    args['psi0'] = states2[-1]
    args['A'] = -1*args['A']
    tlist3, states3 = Ypiby2_square(args, show_plot=False)
    tlist3 = tlist3 + tlist2[-1]
    
    tlist = [*tlist1, *tlist2, *tlist3]
    states = states1 + states2 + states3
    
    if show_plot:
        sm = destroy(2)
        fig, axes = plt.subplots(1, 2, figsize=(16,3))
        axes[0].plot(tlist, expect(sm.dag() * sm, states))
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Avg. photon')
        plot_fock_distribution(states3[-1], fig=fig, ax=axes[1], title="Final state");
        print('Final avg. photon = %f' %(expect(sm.dag() * sm, states3[-1])))
        plot_bloch(states)
    else:
        return tlist, states
    
    
def Ypi_square(args, show_plot=True):
    # Ry[pi]= Rx[-x, r].Rz[pi - z r].Rx[x, r]
    
    def optx(r):
        return math.acos(-r**2)/sqrt(1 + r**2)

    def optz(r):
        return (2/r) * math.atan(r/sqrt(1 - r**2))

    zbyx_rate = args['Xpi']/args['Zpi']/args['A'] # Xpi = pi length when A = 1
    args['x_pulse'] = optx(zbyx_rate) / np.pi * args['Xpi']/args['A']
    args['z_pulse'] = (np.pi - optz(zbyx_rate)*zbyx_rate) / np.pi * args['Zpi']

    if show_plot:
        evolve_fluxonium_square(args,show_plot=show_plot)
    else:
        tlist, states = evolve_fluxonium_square(args,show_plot=show_plot)
        return tlist, states

    
def Ypiby2_triangular(args, show_plot=True):
    # Ry[pi/2]= Rx[-x, r].Rz[z].Rx[x, r], valid for 0<= r <= sqrt(2) - 1 = 0.414

    def optx(r):
        return math.acos(-r*(1+r)/(1-r))/sqrt(1 + r**2)

    def optz(r):
        return 2 * math.atan(sqrt(1-2*r-2*r**3-r**4) / ((1+r)*sqrt(1 + r**2)))

    zbyx_rate = args['Xpi']/args['Zpi']/args['A']
    args['x_pulse'] = optx(zbyx_rate) / pi * args['Xpi']/args['A']
    args['z_pulse'] = optz(zbyx_rate) / pi * args['Zpi']

    if show_plot:
        evolve_fluxonium_triangular(args,show_plot=show_plot)
    else:
        tlist, states = evolve_fluxonium_triangular(args,show_plot=show_plot)
        return tlist, states
#ENDDEF


def evolve_fluxonium_triangular(args, show_plot=False):
    ''' Net zero flux pulse: +ve X, Z, -X pulse '''
    
    sx = sigmax()
    sz = sigmaz()
    sm = destroy(2)

    H0 = + pi/args['Zpi'] * sz/2
    H1 = + pi/args['Xpi'] * sx/2

    x_pulse = args['x_pulse']
    z_pulse = args['z_pulse']
    A = 2*args['A'] # X-drive strength, 2 times to keep similar with a square pulse

    def H1_coeff(t, args):
        # First +ve triangle
        if t<(x_pulse/2):
            return A * 2*t/x_pulse
        elif t>=(x_pulse/2) and t<x_pulse:
            return A * (2 - 2*t/x_pulse)

        # Z-rotation only
        elif t>=x_pulse and t<x_pulse + z_pulse:
            return 0

        # Second -ve triangle
        elif t>=x_pulse + z_pulse and t<3/2*x_pulse + z_pulse:
            return -A * 2*(t - x_pulse - z_pulse)/x_pulse
        else:
            return -A * (2 - 2*(t - x_pulse - z_pulse)/x_pulse)
    
    Ht = [H0, [H1, H1_coeff]]
    
    Tend = 2*x_pulse + z_pulse
    tlist = np.linspace(0, Tend, args['Tpoints'])

    output = mesolve(Ht, psi0, tlist, args['c_ops'], [], args = args)
    
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(16,3))
        axes[0].plot(tlist, expect(sm.dag() * sm, output.states))
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Avg. photon')
        plot_fock_distribution(output.states[-1], fig=fig, ax=axes[1], title="Final state");

        print("X pulse: {:.16f}, Z pulse: {:.16f}, Total: {:.16f}".format(x_pulse,z_pulse,Tend))
        print("Final avg. photon = {:.16f}".format(expect(sm.dag() * sm, output.states[-1])))
        print("final state\n{}".format(output.states[-1]))
        plot_bloch(output.states)
    
    else:
        return tlist, output.states
#ENDDEF
    
def evolve_fluxonium_square(args, show_plot=False):
    ''' Net zero flux pulse: +ve X, Z, -X pulse '''
    
    sx = sigmax()
    sz = sigmaz()
    sm = destroy(2)

    H0 = + np.pi * args['f'] * sz
    H1 = + np.pi * args['a'] * sx

    x_pulse = args['x_pulse']
    z_pulse = args['z_pulse']
    
    def H1_coeff(t, args):
        if t<x_pulse:
            return 1
        elif t>=x_pulse and t<x_pulse + z_pulse:
            return 0
        else:
            return -1
    
    Ht = [H0, [H1, H1_coeff]]
    
    Tend = 2 * args['x_pulse'] + args['z_pulse']
    tlist = np.linspace(0, Tend, args['Tpoints'])

    output = mesolve(Ht, args['psi0'], tlist, args['c_ops'], [], args = args)
    
    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(16,3))
        axes[0].plot(tlist, expect(sm.dag() * sm, output.states))
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Avg. photon')
        plot_fock_distribution(output.states[-1], fig=fig, ax=axes[1], title="Final state");

        print("X pulse: {:.16f}, Z pulse: {:.16f}, Total: {:.16f}".format(x_pulse,z_pulse,Tend))
        print("Final avg. photon = {:.16f}".format(expect(sm.dag() * sm, output.states[-1])))
        print("final state\n{}".format(output.states[-1].full()))
        plot_bloch(output.states)
    
    else:
        return tlist, output.states
#ENDDEF
    
    
def Ypiby2_square(args, show_plot=True):
    # Ry[pi/2]= Rx[-x, r].Rz[z].Rx[x, r], valid for 0<= r <= sqrt(2) - 1 = 0.414

    def optx(r):
        return np.arccos(-r*(1+r)/(1-r))/np.sqrt(1 + r**2)

    def optz(r):
        return 2 * np.arctan(np.sqrt(1-2*r-2*r**3-r**4) / ((1+r)*np.sqrt(1 + r**2)))

    zbyx_rate = args['f']/args['a']
    args['x_pulse'] = optx(zbyx_rate) / (2 * np.pi * args['a'])
    args['z_pulse'] = optz(zbyx_rate) / (2 * np.pi * args['f'])

    if show_plot:
        evolve_fluxonium_square(args,show_plot=show_plot)
    else:
        tlist, states = evolve_fluxonium_square(args,show_plot=show_plot)
        return tlist, states
#ENDDEF


def main():
    psi0 = (basis(2, 0) + basis(2, 1)).unit()
    s0r = 0.45122000269511986
    s0i = 0.5977953610733032j
    s1r = 0.6624293175146164
    s1i = 0.01512001138142209j
    # psi0 = Qobj(np.array([[s0r + s0i],
    #                       [s1r + s1i]]))
    myargs = {
        "psi0": psi0,
        "z_pulse": 1,
        "x_pulse": 1,
        "A": 1,
        "f": 1.4e-2,
        "a": 1.25e-1,
        "Zpi": 35.71428571428572,
        "Xpi": 4,
        "c_ops": [],
        "Tpoints": int(1e4),
    }
    res = Xpiby2_square(myargs, show_plot=False)
    (tlist, states) = res
    initial_state = psi0.full()
    final_state = states[-1].full()
    print("is:\n{}\nfs:\n{}"
          "".format(initial_state, final_state))
#ENDDEF


if __name__ == "__main__":
    main()
#ENDIF
