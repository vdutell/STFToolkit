import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
import matplotlib.colors as colors

# for import plotting.* don't import the things above globally
__all__ = ['da_plot_power',
           'plot_velocity_spec']

def da_plot_power(power_spectrum, fqspace, fqtime, nsamples = 7, minmaxcolors=(None, None), figname='nameless', saveloc='./output', logscale=True, show_onef_line=False):
    '''
    THE Dong & Attick Plot (Dong & Attick 1995; fig 7) with heatmap added.
    Produces three plots:
        - Heatmap displying the joint spatial/temporal amplitude
        - Samples from this plot along spatial and temporal lines:
            * plots of spatial power spectrum along given temporal values
            * plots of temporal spectrum along given spatial values

    Parameters
    ----------
        power_spectrum:   2d numpy array of spatio/temporal power
        fqspace:    1d numpy array of spatial frequency vals for spectrum
        fqtime:   1d numpy array of temporal frequency vals for spectrum
        nsamples: integer number of lines to plot for sampling plots
        power (bool):   Are we calculating the power spectrum (default) or amplitude
        psd (bool):     Normalize to power spectral density (default), or raw power?
        figname (str):  String prefix to name of figure to be saved
    
    Returns:
    --------
        mftchunk
        azmchunk
        freqspace1d
        freqspacefull
        freqtime

    '''
    #calculate sampling positions
    #space
    space_end_offset = 1
    space_end_sample = len(fqspace)-space_end_offset #sample everything for now
    #time
    time_end_offset = 1
    time_end_sample = len(fqtime)-time_end_offset #sample everything for now
    #joint
    n_datpoints = len(fqspace)*len(fqtime)
    
    #colors for lines: generate by indexes in log coordinates.
    spacesamplefqs_idx = np.round(np.geomspace(space_end_offset, space_end_sample,
                                      nsamples),0).astype(int)
    timesamplefqs_idx = np.round(np.geomspace(time_end_offset, time_end_sample,
                                     nsamples),0).astype(int)

    spacecolors = np.array(['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])#[::-1]
    timecolors = spacecolors

    #make a grid
    fig = plt.figure(figsize=(10,6))
    full_grid = gsp.GridSpec(2,3)
    
    #layout of subplots
    grid_hm = gsp.GridSpecFromSubplotSpec(1,1,subplot_spec=full_grid[0:2,0:2])
    grid_time = gsp.GridSpecFromSubplotSpec(1,1,subplot_spec=full_grid[1,2])
    grid_space = gsp.GridSpecFromSubplotSpec(1,1,subplot_spec=full_grid[0,2])
    
    #heatmap
    axes_hm = plt.subplot(grid_hm[0])
    
    power_spectrum = np.log10(power_spectrum)
    
    if(minmaxcolors[0]):
        minc = np.log10(np.abs(minmaxcolors[0]))
        maxc = np.log10(np.abs(minmaxcolors[1]))
    else:
        minc = np.min(power_spectrum)
        maxc = np.max(power_spectrum)

    clev = np.arange(minc,maxc,0.5)
    hm = axes_hm.contourf(fqtime, fqspace, power_spectrum,
                          clev,
                          cmap='gray',
                          norm=mpl.colors.Normalize(minc, maxc))
    
    if(logscale):
        axes_hm.set_xscale("log") 
        axes_hm.set_yscale("log")
    axes_hm.set_xlabel('Hz')
    axes_hm.set_ylabel('cycles/deg')
    axes_hm.set_title(f'{figname} Log Power') 
    plt.colorbar(hm)

    #add lines
    for s in range(nsamples):
        #lines in time
        axes_hm.axvline(fqtime[timesamplefqs_idx[s]],c=timecolors[s],ls='-')
        #lines in space
        axes_hm.axhline(fqspace[spacesamplefqs_idx[s]],c=spacecolors[s],ls='--')

    #spaceplot
    axes_space = plt.subplot(grid_space[0])
    for i, tf_idx in enumerate(timesamplefqs_idx):
        axes_space.semilogx(fqspace, power_spectrum[:,tf_idx],
                        label='{0:0.1f} Hz'.format(fqtime[tf_idx]),
                        c=timecolors[i])
        
    #print(np.log(np.max(power_spectrum)/fqspace))
    #if(show_onef_line):
        #axes_space.semilogx(fqspace,np.max(power_spectrum)/(fqspace),c='black') # 1/f line
        #axes_space.semilogx(fqspace,np.max(power_spectrum)/(fqspace**2),c='black') # 1/f^2 line
    
    axes_space.set_title('Spatial Frequency')
    axes_space.set_xlabel('cpd')
    axes_space.set_ylabel(f'Log Power')
    axes_space.set_xlim(fqspace[1],fqspace[-1])
    axes_space.set_ylim(bottom=minc) #, top=maxc)
    axes_space.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
    axes_space.legend(fontsize=8)

    #timeplot
    axes_time = plt.subplot(grid_time[0])
    for i, sf_idx in enumerate(spacesamplefqs_idx):
        axes_time.semilogx(fqtime, power_spectrum[sf_idx,:],
                       label='{0:0.1f} cpd'.format(fqspace[sf_idx]),
                       c=spacecolors[i])
        
    #if(show_onef_line):    
    #    axes_time.plot(fqtime[1:],1e4/(fqtime[1:]),c='black') # 1/f line
    #    axes_time.plot(fqtime[1:],1e7/(fqtime[1:]**2),c='black') # 1/f^2 line
        
    axes_time.set_title('Temporal Frequency')
    axes_time.set_xlabel('Hz')
    axes_time.set_ylabel(f'Log Power')
    axes_time.set_xlim(fqtime[1],fqtime[-1])
    axes_time.set_ylim(bottom=minc) #, top= maxc)
    axes_time.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
    axes_time.legend(fontsize=8)

    plt.tight_layout()
    
    plt.savefig(f'{saveloc}/Power_{figname}.png')

    
def plot_velocity_spec(velocity_vals, velocity_spec, label):
    """
    Plotter for velocity spectra
    
    Parameters:
        velocity spectrum: 1d numpy array of velocity values (bins)
        velocity vals: 1d numpy array mean amplitude at each velocity value (size must match velocity_spectrum)
        label: label for plot
        
    Returns:
        figure: matplotlib figure plotted
    """

    p = plt.loglog(velocity_vals,
             velocity_spec,
             'o-',
             label=label)

    plt.legend()
    plt.xlabel('Velocity (degrees/sec)')
    plt.ylabel('Mean Power')
    plt.title('Movie Velocity Spectrum')
    
    return(p)