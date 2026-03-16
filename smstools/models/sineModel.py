"""Sinusoidal model for sound analysis and synthesis.

Provides frame-by-frame and streaming analysis/synthesis using a sinusoidal model
with peak-based spectral representation. Includes sinusoid tracking across frames
and track-cleaning utilities for robust long-term sinusoidal modeling.

Key functions:
  - sineModel: One-pass analysis/synthesis without tracking
  - sineModelAnal: Multi-frame analysis with automatic track continuity
  - sineModelSynth: Reconstruct audio from sinusoidal tracks
  - sineTracking: Assign current peaks to previous tracks
  - cleaningSineTracks: Remove short track fragments
"""

import math

import numpy as np
from scipy.fft import fftshift, ifft
from scipy.signal.windows import blackmanharris, triang

from smstools.models import dftModel as DFT
from smstools.models import utilFunctions as UF


def sineTracking(pfreq, pmag, pphase, tfreq, freqDevOffset=20, freqDevSlope=0.01):
    """
    Assign spectral peaks to existing sinusoidal tracks from previous frame.
    
    Uses frequency proximity and magnitude ordering to maintain track continuity
    across analysis frames.
    
    Args:
        pfreq: Peak frequencies in current frame
        pmag: Peak magnitudes in current frame
        pphase: Peak phases in current frame
        tfreq: Track frequencies from previous frame
        freqDevOffset: Frequency deviation tolerance at 0 Hz (Hz)
        freqDevSlope: Frequency deviation slope (frequency-dependent tolerance)
    
    Returns:
        tfreqn, tmagn, tphasen: Updated track frequencies, magnitudes, phases
    """

    tfreqn = np.zeros(tfreq.size)
    tmagn = np.zeros(tfreq.size)
    tphasen = np.zeros(tfreq.size)
    
    pindexes = np.nonzero(pfreq)[0]
    incomingTracks = np.nonzero(tfreq)[0]
    newTracks = np.full(tfreq.size, -1, dtype=int)
    magOrder = np.argsort(-pmag[pindexes])
    
    pfreqt = np.copy(pfreq)
    pmagt = np.copy(pmag)
    pphaset = np.copy(pphase)
    used_peak_mask = np.zeros(pfreq.size, dtype=bool)
    available_tracks = list(incomingTracks)

    # Assign peaks to incoming tracks (sorted by magnitude)
    for i in magOrder:
        if not available_tracks:
            break
        peak_idx = pindexes[i]
        track_distances = np.abs(pfreqt[peak_idx] - tfreq[available_tracks])
        closest_pos = np.argmin(track_distances)
        track_idx = available_tracks[closest_pos]
        freqDistance = abs(pfreq[peak_idx] - tfreq[track_idx])
        if freqDistance < (freqDevOffset + freqDevSlope * pfreq[peak_idx]):
            newTracks[track_idx] = peak_idx
            used_peak_mask[peak_idx] = True
            available_tracks.pop(closest_pos)
    
    # Transfer assigned tracks
    assigned_idx = np.nonzero(newTracks != -1)[0]
    tfreqn[assigned_idx] = pfreqt[newTracks[assigned_idx]]
    tmagn[assigned_idx] = pmagt[newTracks[assigned_idx]]
    tphasen[assigned_idx] = pphaset[newTracks[assigned_idx]]
    
    # Get unassigned peaks, sorted by magnitude
    peaksleft_idx = np.nonzero((~used_peak_mask) & (pfreq > 0))[0]
    if peaksleft_idx.size > 0:
        peaksleft = peaksleft_idx[np.argsort(-pmagt[peaksleft_idx])]
    else:
        peaksleft = np.array([], dtype=int)
    
    # Fill empty tracks with remaining peaks
    emptyt = np.nonzero(tfreq == 0)[0]
    if peaksleft.size > 0 and emptyt.size >= peaksleft.size:
        tfreqn[emptyt[:peaksleft.size]] = pfreqt[peaksleft]
        tmagn[emptyt[:peaksleft.size]] = pmagt[peaksleft]
        tphasen[emptyt[:peaksleft.size]] = pphaset[peaksleft]
    elif peaksleft.size > 0 and emptyt.size < peaksleft.size:
        tfreqn[emptyt] = pfreqt[peaksleft[:emptyt.size]]
        tmagn[emptyt] = pmagt[peaksleft[:emptyt.size]]
        tphasen[emptyt] = pphaset[peaksleft[:emptyt.size]]
        tfreqn = np.append(tfreqn, pfreqt[peaksleft[emptyt.size:]])
        tmagn = np.append(tmagn, pmagt[peaksleft[emptyt.size:]])
        tphasen = np.append(tphasen, pphaset[peaksleft[emptyt.size:]])
    
    return tfreqn, tmagn, tphasen


def cleaningSineTracks(tfreq, minTrackLength=3):
    """
    Remove sinusoidal track fragments below minimum duration threshold.
    
    Zeros out track segments (contiguous non-zero regions) shorter than the
    specified minimum track length.
    
    Args:
        tfreq: 2D array of track frequencies (frames x tracks)
        minTrackLength: Minimum number of consecutive frames for track retention
    
    Returns:
        tfreq: Modified track array with short fragments removed
    """

    if tfreq.shape[1] == 0:  # if no tracks return input
        return tfreq
    nFrames = tfreq[:, 0].size  # number of frames
    nTracks = tfreq[0, :].size  # number of tracks in a frame
    for t in range(nTracks):  # iterate over all tracks
        trackFreqs = tfreq[:, t]  # frequencies of one track
        trackBegs = (
            np.nonzero(
                (trackFreqs[: nFrames - 1] <= 0)  # begining of track contours
                & (trackFreqs[1:] > 0)
            )[0]
            + 1
        )
        if trackFreqs[0] > 0:
            trackBegs = np.insert(trackBegs, 0, 0)
        trackEnds = (
            np.nonzero(
                (trackFreqs[: nFrames - 1] > 0)  # end of track contours
                & (trackFreqs[1:] <= 0)
            )[0]
            + 1
        )
        if trackFreqs[nFrames - 1] > 0:
            trackEnds = np.append(trackEnds, nFrames - 1)
        trackLengths = 1 + trackEnds - trackBegs  # lengths of trach contours
        for i, j in zip(trackBegs, trackLengths):  # delete short track contours
            if j <= minTrackLength:
                trackFreqs[i : i + j] = 0
    return tfreq


def _build_synthesis_window(N, H):
    """
    Construct normalized synthesis window for overlap-add processing.
    
    Combines triangular and Blackman-Harris windows for smooth spectral
    reconstruction and minimal spectral artifacts.
    
    Args:
        N: FFT size
        H: Hop size
    
    Returns:
        sw: Normalized synthesis window of length N
    """
    hN = N // 2
    sw = np.zeros(N)
    ow = triang(2 * H)
    sw[hN - H : hN + H] = ow
    bh = blackmanharris(N)
    bh = bh / np.sum(bh)
    sw[hN - H : hN + H] = sw[hN - H : hN + H] / bh[hN - H : hN + H]
    return sw


def sineModel(x, fs, w, N, t):
    """
    Single-pass sinusoidal analysis/synthesis without frame tracking.
    
    Analyzes overlapping frames to extract peaks, synthesizes spectral sines,
    and reconstructs via overlap-add. Best for real-time or one-pass processing.
    
    Args:
        x: Input audio signal
        fs: Sampling rate (Hz)
        w: Analysis window
        N: FFT size
        t: Peak detection threshold (negative dB)
    
    Returns:
        y: Resynthesized audio signal
    """

    hM1 = (w.size + 1) // 2
    hM2 = w.size // 2
    Ns = 512
    H = Ns // 4
    hNs = Ns // 2
    pin = max(hNs, hM1)
    pend = x.size - max(hNs, hM1)
    yw = np.zeros(Ns)
    y = np.zeros(x.size)
    w = w / np.sum(w)
    sw = _build_synthesis_window(Ns, H)
    while pin < pend:
        # Analysis: extract peaks from frame
        x1 = x[pin - hM1 : pin + hM2]
        mX, pX = DFT.dftAnal(x1, w, N)
        ploc = UF.peakDetection(mX, t)
        iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)
        ipfreq = fs * iploc / float(N)
        
        # Synthesis: generate spectral sines and overlap-add
        Y = UF.genSpecSines(ipfreq, ipmag, ipphase, Ns, fs)
        fftbuffer = np.real(ifft(Y))
        yw[: hNs - 1] = fftbuffer[hNs + 1 :]
        yw[hNs - 1 :] = fftbuffer[: hNs + 1]
        y[pin - hNs : pin + hNs] += sw * yw
        pin += H
    return y


def sineModelAnal(x, fs, w, N, H, t, maxnSines=100, minSineDur=0.01, freqDevOffset=20, freqDevSlope=0.01):
    """
    Multi-frame sinusoidal analysis with automatic track continuity.
    
    Extracts spectral peaks from successive frames and maintains sinusoidal
    tracks across frames using proximity-based matching. Short track fragments
    are removed based on minimum duration.
    
    Args:
        x: Input audio signal
        fs: Sampling rate (Hz)
        w: Analysis window
        N: FFT size
        H: Hop size (samples)
        t: Peak detection threshold (negative dB)
        maxnSines: Maximum tracks per frame
        minSineDur: Minimum track duration (seconds)
        freqDevOffset: Frequency deviation tolerance at 0 Hz (Hz)
        freqDevSlope: Frequency-dependent deviation slope
    
    Returns:
        xtfreq, xtmag, xtphase: 2D arrays of track parameters (frames x tracks)
    """

    if minSineDur < 0:
        raise ValueError("Minimum duration of sine tracks smaller than 0")

    hM1 = (w.size + 1) // 2
    hM2 = w.size // 2
    x = np.concatenate([np.zeros(hM2), x, np.zeros(hM2)])
    pin = hM1
    pend = x.size - hM1
    w = w / np.sum(w)
    tfreq = np.array([])
    frames_list = []
    
    while pin < pend:
        # Extract and interpolate spectral peaks
        x1 = x[pin - hM1 : pin + hM2]
        mX, pX = DFT.dftAnal(x1, w, N)
        ploc = UF.peakDetection(mX, t)
        iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)
        ipfreq = fs * iploc / float(N)
        
        # Track sinusoid continuity across frames
        tfreq, tmag, tphase = sineTracking(
            ipfreq, ipmag, ipphase, tfreq, freqDevOffset, freqDevSlope
        )
        
        # Limit to maximum number of sines and pad to fixed size
        n_sines = min(maxnSines, tfreq.size)
        jtfreq = np.zeros(maxnSines)
        jtmag = np.zeros(maxnSines)
        jtphase = np.zeros(maxnSines)
        jtfreq[:n_sines] = tfreq[:n_sines]
        jtmag[:n_sines] = tmag[:n_sines]
        jtphase[:n_sines] = tphase[:n_sines]
        
        frames_list.append((jtfreq, jtmag, jtphase))
        pin += H
    
    # Stack frames into 2D arrays
    if frames_list:
        xtfreq = np.array([f[0] for f in frames_list])
        xtmag = np.array([f[1] for f in frames_list])
        xtphase = np.array([f[2] for f in frames_list])
    else:
        xtfreq = np.array([]).reshape(0, maxnSines)
        xtmag = np.array([]).reshape(0, maxnSines)
        xtphase = np.array([]).reshape(0, maxnSines)
    # Remove tracks shorter than minimum duration
    minTrackFrames = round(fs * minSineDur / H)
    xtfreq = cleaningSineTracks(xtfreq, minTrackFrames)
    return xtfreq, xtmag, xtphase


def sineModelSynth(tfreq, tmag, tphase, N, H, fs):
    """
    Reconstruct audio from sinusoidal track parameters.
    
    Generates spectral sine components for each track and synthesis frame,
    then reconstructs via overlap-add with spectral phase propagation between
    frames for coherent reconstruction.
    
    Args:
        tfreq: 2D array of track frequencies (frames x tracks)
        tmag: 2D array of track magnitudes (frames x tracks)
        tphase: 2D array of track phases (frames x tracks)
        N: Synthesis FFT size
        H: Hop size (samples)
        fs: Sampling rate (Hz)
    
    Returns:
        y: Reconstructed audio signal
    """

    hN = N // 2
    L = tfreq.shape[0]
    pout = 0
    ysize = H * (L + 3)
    y = np.zeros(ysize)
    sw = _build_synthesis_window(N, H)
    lastytfreq = tfreq[0, :]
    ytphase = 2 * np.pi * np.random.rand(tfreq[0, :].size)
    for l in range(L):
        # Use provided phases or propagate from previous frame
        if tphase.size > 0:
            ytphase = tphase[l, :]
        else:
            ytphase += (np.pi * (lastytfreq + tfreq[l, :]) / fs) * H
        
        # Generate spectral sines and synthesize frame
        Y = UF.genSpecSines(tfreq[l, :], tmag[l, :], ytphase, N, fs)
        lastytfreq = tfreq[l, :]
        ytphase = ytphase % (2 * np.pi)
        yw = np.real(fftshift(ifft(Y)))
        y[pout : pout + N] += sw * yw
        pout += H
    
    # Trim half-window padding from start and end
    y = y[hN : y.size - hN]
    return y
