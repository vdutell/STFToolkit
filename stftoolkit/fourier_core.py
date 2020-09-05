import numpy as np
import cupy as cp
import time

#support for multiprocessing to speed up azimuthal average binning 
import multiprocessing
from joblib import Parallel, delayed
num_cores = multiprocessing.cpu_count()


# for import plotting.* don't import the things above globally
__all__ = ['cos_filter_3d',
           'azimuthalAverage',
           'cubify',
           'st_ps',
           'calc_velocity_spec']

def cos_filter_3d(movie):
    #filter first with cosine window
    (dimt,dim1,dim2) = np.shape(movie)
    dt = np.tile(np.hanning(dimt),(dim1,dim2,1)).transpose(2,0,1)
    d1 = np.tile(np.hanning(dim1),(dimt,dim2,1)).transpose(0,2,1)
    d2 = np.tile(np.hanning(dim2),(dimt,dim1,1))

    cosfilter = dt*d1*d2
    movie = np.array(cosfilter * movie)
    return(movie)

def mask_angle_wedge(direction_angle, indices, width=np.pi/4):
    '''
    Helper Fucntion for aximuthalAverage for directional support
    Create a Mask on the indices centered at direction_angle, with given width (default 1/8 of circle)
    Arguments:
        direction_angle (float): angle in radians (based on unit circle with negative values in bottom two quadrants)
        indices (2d array): indices of axes, with (0,0) at center
        width (float): width in radians for wedge (default is 1/8 or circle of pi/4)
    Returns:
        keep_angles_mask (2d boolean): mask on indices indicating which incides fall inside angle, which do not
    '''
    
    start_angle = direction_angle-(width/2)
    end_angle = direction_angle+(width/2)
    
    angles = np.arctan2(indices[1],indices[0])
    
    #if we are on the edge of the +/- edge at pi, do special handling
    if end_angle > np.pi:
        end_angle = end_angle - 2*np.pi
        keep_angles_mask = (angles>=start_angle) | (angles<end_angle)
    elif start_angle < -np.pi:
        end_angle = end_angle + 2*np.pi
        keep_angles_mask = (angles>=start_angle) | (angles<end_angle)
    else:
        keep_angles_mask = (angles>=start_angle) * (angles<end_angle)
            
    return(keep_angles_mask)


def azimuthalAverage(image, nyquist, angles='all', center=None, bin_in_log=False, return_fqs=True):
    """      
    Calculate the azimuthally averaged radial profile. (Intended for 2d Power Spectra)
    image - The 2D image (2d power spectrum)
    nyquist - max frequency value (assume same for x and y)
    angles - the section of spatial frequency to return (all, vert, horiz, r_diag, l_diag)
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)
    num_bins = np.min(image.shape)//2

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    #ASSUME HERE THAT MAX FREQUENCY IS EQUAL ON BOTH AXES & GRANULARITY VARIES ***
    normalized = ((x-center[0])/np.max(x),(y-center[1])/np.max(y))
    r = np.hypot(normalized[0], normalized[1])
    #don't calculate corners
    keep_circle = abs(r)<=np.max(normalized)

    #keep only the angles we are interested in (all, vertical, horizontal, r_diag, l_diag)
    if(angles=='all'):
        wedge_mask = np.full(np.shape(r), True)
    elif(angles=='vert'):
        angle_1 = np.pi/2
        angle_2 = -np.pi/2
        wedge_1 = mask_angle_wedge(angle_1, normalized)
        wedge_2 = mask_angle_wedge(angle_2, normalized)
        wedge_mask = wedge_1 | wedge_2
    elif(angles=='horiz'):
        angle_1 = 0
        angle_2 = np.pi
        wedge_1 = mask_angle_wedge(angle_1, normalized)
        wedge_2 = mask_angle_wedge(angle_2, normalized)
        wedge_mask = wedge_1 | wedge_2
    elif(angles=='r_diag'):
        angle_1 = np.pi/4
        angle_2 = -3*np.pi/4
        wedge_1 = mask_angle_wedge(angle_1, normalized)
        wedge_2 = mask_angle_wedge(angle_2, normalized)
        wedge_mask = wedge_1 | wedge_2
    elif(angles=='l_diag'):
        angle_1 = 3*np.pi/4
        angle_2 = -np.pi/4
        wedge_1 = mask_angle_wedge(angle_1, normalized)
        wedge_2 = mask_angle_wedge(angle_2, normalized)
        wedge_mask = wedge_1 | wedge_2
    
    #combine all the values we want to keep and retain only those in image and radius
    keep_values = keep_circle & wedge_mask
    r = r[np.where(keep_values)]
    image = image[keep_values]

    # number of bins should be equivalent to the number of bins along the shortest axis of the image.
    if(bin_in_log):
        bin_edges = np.histogram_bin_edges(np.log(r), num_bins)
        bin_edges = np.exp(bin_edges)
    else:
        bin_edges = np.histogram_bin_edges(r, num_bins)
        
    bin_centers = bin_edges[:-1] + ((bin_edges[1]-bin_edges[0])/2)
    bin_centers = bin_centers/np.max(bin_centers)*nyquist

    bin_edges[0] -= 1
    bin_edges[-1] += 1
    
    r_binned = np.digitize(r, bin_edges) - 1
    binmean = np.zeros(num_bins)
    for i in range(num_bins):
        binmean[i] = np.mean(image[np.where(r_binned==(i))])

    if(return_fqs):
        return(binmean, bin_centers)
    else:
        return(binmean)

def cubify(arr, newshape):
    '''
        chunks array (movie) into equal smaler cubes.
        Taken directly From: https://stackoverflow.com/questions/42297115/numpy-split-cube-into-cubes
        
    '''
    oldshape = arr.shape
    
    #make sure we aren't asking for longer movies than we started with!
    assert oldshape[0]>=newshape[0], f"desired movie length is longer than original: {oldshape[0]} < {newshape[0]}!"
    
    repeats = (np.array(oldshape) / np.array(newshape)).astype(int)
    
    tmpshape = np.column_stack([repeats, newshape]).astype('int').ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    
    # newshape must divide oldshape evenly or else ValueError will be raised
    new = arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)
    return new


def st_ps(movie, ppd=1, fps=1, cosine_window=True, rm_dc=False, bin_in_log=False, use_cupy_fft=True, parallelize_azmaverage=True):
    '''
    Calculate the spatiotemporal power spectrum of a movie.
    
    Parameters
    ----------
    movie:      list of 3d numpy arrays definig movie for 3d fourier transform analysis. Must be in shape (frames, xpixels, ypixels)
    ppd:        pixels per degree of frames
    fps:        frames per second of movie capture
    cosine_window: Use a cosine window to negate edge effects?
    rm_dc:      remove dc component before Fourier transform?
    
    Returns:
    --------
    ps_3D:      3d (full) power specturm of movie
    ps_2D:      list of 2d (spatially averaged) power specturms of movie [all_angles, vert, horiz, left_diag, right_diag]
    fq1d:       spatial frequency spectrum (dimensions of 0 axis of ps)
    ft1d:       temporal frequency specturm (dimensions of 1 axis of ps)
    
    '''
    

    #remove color channel if needed
    if(len(np.shape(movie)) > 3):
        movie = np.mean(movie,axis=-1)
    (dimf, dim1, dim2) = np.shape(movie)
    
    #raised cosyne window on image to avoid border artifacts
    if(cosine_window):
        movie = cos_filter_3d(movie)
    
    #subtract DC component
    if(rm_dc):
        movie = movie - np.mean(movie)    
    
    #3d ft
    #option to use GPU accellerated
    if(use_cupy_fft):
        ps_3d = cp.asnumpy(cp.fft.fftshift(cp.abs(cp.fft.fftn(cp.asarray(movie))**2)))
    else:
        ps_3d = np.fft.fftshift(np.abs(np.fft.fftn(movie))**2)
    
    fqs_time = np.fft.rfftfreq(np.shape(movie)[0])
    
    #azimuthal avearge over spatial dimension at each temporal frequency
    #we do this for multi8ple different angle sections [all, vert, horizl l_diag, r_diag]
    angles = ['all','vert','horiz','l_diag','r_diag']
    ps_2ds = [[] for _ in range(len(angles))]
        
    if(parallelize_azmaverage):
        #Parallel implemetation
        for i, angle in enumerate(angles):
            ps = Parallel(n_jobs=num_cores)(delayed(azimuthalAverage)(ps_3d[len(fqs_time)-2+f,:,:], max(fqs_time), angles=angle, bin_in_log=bin_in_log, return_fqs=False) for f in range(len(fqs_time)))
            ps_2ds[i].append(ps)
            #RUN AN EXTRA TIME TO GET FREQUENCIES
            _, fqs_space = azimuthalAverage(ps_3d[-1,:,:], max(fqs_time), angles=angle, bin_in_log=bin_in_log)
        ps_2ds = [np.array(ps_2d[0]).T for ps_2d in ps_2ds] #spatial as first dim, temporal as second

    else:
        #NON Parallel implemetation
        for i, angle in enumerate(angles):
            for f in range(len(fqs_time)):
                #take only the second half (real part)
                ps, fqs_space = azimuthalAverage(ps_3d[len(fqs_time)-2+f,:,:], max(fqs_time), angles=angle,bin_in_log=bin_in_log)
                ps_2ds[i].append(ps)
                 
        ps_2ds = [np.array(ps_2d).T for ps_2d in ps_2ds] #spatial as first dim, temporal as second


    fqs_space = fqs_space*ppd
    fqs_time = fqs_time*fps

    if(rm_dc):
        ps_3d = ps_3d[1:, 1:, 1:]
        ps_2ds = [ps[1:, 1:] for ps in ps_2ds]
        fqs_space = fqs_space[1:]
        fqs_time = fqs_time[1:]
    
    return(ps_3d, ps_2ds, fqs_space, fqs_time)

def calc_velocity_spec(spectrum, fqspace, fqtime, nbins=20, bin_in_log=False):
    """
    A spatiotemporal power spectrum has a corresponding velocity spectrum.
    This is calculated by dividing the temporal frequencies by the spatial frequencies.
    This function converts a joint spatiotemporal amplitude spectrum into a 1D velocity spectrum.
    
    Parameters:
    spectrum: 2d numpy array of spatio/temporal amlpitude spectrum
    fqspace: 1d numpy array defining spatial frequencies of spectrum
    fqtime: 1d numpy array defining temporal frequencies of spectrum
    nbins: integer number of bins in which to group velocity values
    bin_in_log: should we perform binning in log space?
    
    Returns:
    bins: 1d numpy array defining bins
    v_spectrum: 1d numpy array of velocity amplitdue spectrum (mean logvelocity amplitude in bin)
    
    """
    
    #remove dc
    fqtime = fqtime[1:]
    fqspace = fqspace[1:]
    spectrum = spectrum[1:,1:]
    
    xx, yy = np.meshgrid(fqtime, fqspace) #remove dc
    if(bin_in_log):
        v = np.log10(xx/yy) #bin velocities in log space
    else:
        v = xx/yy
        
    counts, bins = np.histogram(v.flatten(), bins=nbins)
    
    spectrum_flat = spectrum.flatten()

    mask = np.array([np.digitize(v.flatten(), bins) == i for i in range(nbins+1)])
    
    v_spectrum = [np.mean(spectrum_flat[i]) for i in mask]
    
    #move back to true velocity bin values (undo our log)
    if(bin_in_log):
        bins = np.array([10**b for b in bins])
    
    return(bins, v_spectrum)