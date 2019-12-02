import numpy as np

# for import plotting.* don't import the things above globally
__all__ = ['average_3d_ps']

def average_3d_ps(movies, chunkshape, fps, ppd):

    """ 
    Calculate the mean power spectrum for a list of many movies
    
    Parameters
    ----------
    movies:      list of 3d numpy arrays defining movies for 3d fourier transform analysis.
    chunklen:   integer number of frames defining the length of movie 'chunks'
    fps:        frame rate of movie
    ppd:        pixels per degree of movie
    
    Returns:
    --------
    psmean:     mean spatiotemporal fourier transform
    fq1d:       spatial frequency spectrum (dimensions of 0 axis of psmean)
    fqt:        temporal frequency spectrum (dimensions of 1 axis of psmean)
    
    """
    
    #keep a mean
    nmovies = len(movies)
    pssum = 0
    
    #ftspecs = []
    for movie in movies:
        
        ps, fq1d, fqt = st_ps(movie, chunkshape, fps, ppd)
        #print(ft.shape, fq1d.shape, fqt.shape)
        #ftspecs.append(ft)
        pssum = ftsum + ps
    
    psmean = pssum / nmovies
    #print(len(ftspecs), ftspecs[1].shape)
    #ftmean = np.mean(np.array(ftspecs),axis=0)
    #print(ftmean.shape)
    return(psmean, fq1d, fqt)