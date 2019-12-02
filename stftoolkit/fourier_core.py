import numpy as np

# for import plotting.* don't import the things above globally
__all__ = ['cos_filter_3d',
           'azimuthalAverage',
           'cubify',
           'st_ps']

def cos_filter_3d(movie):
    #filter first with cosine window
    (dimt,dim1,dim2) = np.shape(movie)
    dt = np.tile(np.hanning(dimt),(dim1,dim2,1)).transpose(2,0,1)
    d1 = np.tile(np.hanning(dim1),(dimt,dim2,1)).transpose(0,2,1)
    d2 = np.tile(np.hanning(dim2),(dimt,dim1,1))

    cosfilter = dt*d1*d2
    movie = np.array(cosfilter * movie)
    return(movie)


def azimuthalAverage(image, nyquist, center=None, bin_in_log=False):
    """      
    Calculate the azimuthally averaged radial profile. (Intended for 2d Power Spectra)
    image - The 2D image (2d power spectrum)
    nyquist - max frequency value (assume same for x and y)
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)
    num_bins = np.min(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    #ASSUME HERE THAT MAX FREQUENCY IS EQUAL ON BOTH AXES & GRANULARITY VARIES ***
    normalized = ((x-center[0])/np.max(x),(y-center[1])/np.max(y))
    r = np.hypot(normalized[0], normalized[1])
    #don't calculate corners
    keep_circle = np.where(r<=np.max(y))
    r = r[keep_circle]
    image = image[keep_circle]

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

    return(binmean, bin_centers)


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


def st_ps(movie, ppd=1, fps=1, cosine_window=True, rm_dc=True):
    '''
    Calculate the spatiotemporal power spectrum of a movie.
    
    Parameters
    ----------
    movie:      list of 3d numpy arrays definig movie for 3d fourier transform analysis.
    ##ppd:        pixels per degree of frames
    ##fps:        frames per second of movie capture
    
    Returns:
    --------
    ps_3D:      3d (full) power specturm of movie
    ps_2D:      2d (spatially averaged) power specturm of movie
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
    ps_3d = np.fft.fftshift(np.abs(np.fft.fftn(movie))**2)
    ps_3d = np.array(ps_3d[np.shape(ps_3d)[0]//2-1:])
    fqs_time = np.fft.rfftfreq(np.shape(movie)[0])
    
    #azimuthal avearge over spatial dimension at each temporal frequency
    ps_2d = []
    for f in range(np.shape(ps_3d)[0]):
        ps, fqs_space = azimuthalAverage(ps_3d[f], max(fqs_time))
        ps_2d.append(ps)
    ps_2d = np.array(ps_2d)
    #fqs_space = np.fft.fftfreq(np.shape(ps_3d))
    #fqs_2d = np.array((fqs_3d[0], fqs_2d[0]))
    
    if(rm_dc):
        ps_3d = ps_3d[1:, 1:, 1:]
        ps_2d = ps_2d[1:, 1:]
        fqs_space = fqs_space[1:]*ppd
        fqs_time = fqs_time[1:]*fps
    
    return(ps_3d, ps_2d, fqs_space, fqs_time)