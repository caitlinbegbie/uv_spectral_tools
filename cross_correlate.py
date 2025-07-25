import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import pearsonr
from scipy.interpolate import interp1d

#-----------------------------------------------------------------------------------------

def linearize(flux, wave, dispersion=None ):
    """ Return flux and wave arrays with a linear wavelength scale
        """

    interp_func = interp1d( wave, flux, 1, bounds_error=False, fill_value=0)
    if not dispersion:
        dispersion = np.median( (wave - np.roll(wave,1)) )
    out_wave = np.arange( wave.min(), wave.max(), dispersion )
    out_flux = interp_func( out_wave )

    return out_flux, out_wave

#-----------------------------------------------------------------------------------------

def calculate_coeff(a, b, shift):
    # for correlation coefficent
    if shift > 0:
        a = a[shift:]
        b = b[:-shift]
    elif shift < 0:
        a = a[:shift]
        b = b[-shift:]
    # else if shift == 0 then we just use a and b as is
    corr_coeff = abs(pearsonr(a, b)[0])

    return corr_coeff

#-----------------------------------------------------------------------------------------

def quad_fit(c, minpix=5, maxpix=5):

    if len(c) == 1:
        return None
    x = np.arange(len(c))
    if np.argmax(c)-minpix > 0:
        x = x[np.argmax(c)-minpix : np.argmax(c)+(maxpix+1)]
        c2 = c[np.argmax(c)-minpix : np.argmax(c)+(maxpix+1)]
    else:
        x = x[0 : np.argmax(c)+(maxpix+1)]
        c2 = c[0 : np.argmax(c)+(maxpix+1)]
    try:
        quad_fit_func = np.poly1d(np.polyfit(x, c2, 2))
        new_shift = (-quad_fit_func[1]/(2*quad_fit_func[2])) # zero point -b/2a
    except ValueError:
        import pdb; pdb.set_trace()

    #plt.plot(x, quad_fit_func(x))
    #plt.plot(new_shift, np.max(c), '^', color='blue', alpha=0.5)

    return new_shift

#-----------------------------------------------------------------------------------------

def direct_correlate(a, b, fit_peak):

    #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    #ax1.plot(a)
    #ax1.plot(b)

    # direct correlation
    c = np.correlate(a, b, mode='full')
    shift = np.argmax(c)

    #ax2.plot(c)
    #ax2.plot(shift, np.max(c), 's', color='red', alpha=0.5)

    if np.isnan(c).any():
        return None, None, None, None
    if fit_peak:
        shift = quad_fit(c)
    if shift == None:
        return None, None, None, None

    er = abs(shift - np.argmax(c))
    # the len(a) and len(b) need to be the same or this will be wrong
    shift = shift - (len(a)-1)
    corr_coeff = calculate_coeff(a, b, int(round(shift)))

    #fig.suptitle('Shift:{} CC:{} Error:{}'.format(shift, corr_coeff, er))
    #plt.show()

    return shift, c, corr_coeff, er

#-----------------------------------------------------------------------------------------

def cross_correlate(flux_a, flux_b, wave_a=None, wave_b=None, subsample=1, fit_peak=True):
    """ Cross correlate two spectra in wavelength space
        Flux A is the reference spectra
        """
    dispersion = np.median( (wave_a - np.roll(wave_a,1)) )
    if dispersion == 0:
        raise ValueError('Dispersion needs to be GT 0')
    #sub sample
    dispersion /= subsample

    low_wave = max(np.nanmin(wave_a), np.nanmin(wave_b))
    high_wave = min(np.nanmax(wave_a), np.nanmax(wave_b))

    index = np.where( (wave_a <= high_wave) & (wave_a >= low_wave) )[0]
    flux_a = flux_a[ index ]
    wave_a = wave_a[ index ]

    index = np.where( (wave_b <= high_wave) & (wave_b >= low_wave) )[0]
    flux_b = flux_b[ index ]
    wave_b = wave_b[ index ]

    flux_a, wave_a = linearize( flux_a, wave_a, dispersion )
    flux_b, wave_b = linearize( flux_b, wave_b, dispersion )

    # formally the function "correlations" below to do the actual cross-correlation:
    lims = (0, 1)

    if len(flux_a) > len(flux_b):
        alims = (0, -1* (1 + len(flux_a) - len(flux_b) ) )

    if len(flux_b) > len(flux_a):
        blims = (0, -1 * (1 + len(flux_b) - len(flux_a) ) )

    # need to normalize this way
    flux_a = (flux_a-flux_a.mean())/flux_a.std()
    flux_b = (flux_b-flux_b.mean())/flux_b.std()

    # redefine the array lengths so the array sizes match
    #flux_a = flux_a[lims[0]:lims[1]]
    #flux_b = flux_b[lims[0]:lims[1]]

    shift, c, corr_coeff, er = direct_correlate(flux_a, flux_b, fit_peak)
    if shift == None:
        return None, None, None

    # converting the shift back to wavelength
    shift = shift * dispersion

    return shift, corr_coeff, er