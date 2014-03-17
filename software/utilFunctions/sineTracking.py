import numpy as np

def cleaningSineTracks(tfreq, minTrackLength=3):
  # delete short sinusoidal tracks 
  # tfreq: frequency of tracks
  # minTrackLength: minimum duration of tracks in number of frames
  # returns tfreqn: frequency of tracks
  nFrames = tfreq[:,0].size         # number of frames
  nTracks = tfreq[0,:].size         # number of tracks in a frame
  for t in range(nTracks):          # iterate over all tracks
    trackFreqs = tfreq[:,t]         # frequencies of one track
    trackBegs = np.nonzero((trackFreqs[:nFrames-1] <= 0) # begining of track contours
                & (trackFreqs[1:]>0))[0] + 1
    if trackFreqs[0]>0:
      trackBegs = np.insert(trackBegs, 0, 0)
    trackEnds = np.nonzero((trackFreqs[:nFrames-1] > 0)  # end of track contours
                & (trackFreqs[1:] <=0))[0] + 1
    if trackFreqs[nFrames-1]>0:
      trackEnds = np.append(trackEnds, nFrames-1)
    trackLengths = 1 + trackEnds - trackBegs             # lengths of trach contours
    for i,j in zip(trackBegs, trackLengths):             # delete short track contours
      if j <= minTrackLength:
        trackFreqs[i:i+j] = 0
  return tfreq

def sineTracking(pfreq, pmag, pphase, tfreq, freqDevOffset=20, freqDevSlope=0.01):
  # tracking sinusoids from one frame to the next
  # pfreq, pmag, pphase: frequencies and magnitude of current frame
  # tfreq: frequencies of incoming tracks
  # freqDevOffset: minimum frequency deviation at 0Hz 
  # freqDevSlope: slope increase of minimum frequency deviation
  # returns tfreqn, tmagn, tphasen: frequencies, magnitude and phase of tracks
  tfreqn = np.zeros(tfreq.size)                              # initialize array for output frequencies
  tmagn = np.zeros(tfreq.size)                               # initialize array for output magnitudes
  tphasen = np.zeros(tfreq.size)                             # initialize array for output phases
  pindexes = np.array(np.nonzero(pfreq), dtype=np.int)[0]    # indexes of current peaks
  incomingTracks = np.array(np.nonzero(tfreq), dtype=np.int)[0] # indexes of incoming tracks
  newTracks = np.zeros(tfreq.size, dtype=np.int) -1           # initialize to -1 new tracks
  magOrder = np.argsort(-pmag[pindexes])                      # order current peaks by magnitude
  pfreqt = np.copy(pfreq)                                     # copy current peaks to temporary array
  pmagt = np.copy(pmag)                                       # copy current peaks to temporary array
  pphaset = np.copy(pphase)                                   # copy current peaks to temporary array

  # continue incoming tracks
  if incomingTracks.size > 0:                                   # if incoming tracks exist
  	for i in magOrder:                                        # iterate over current peaks
  		if incomingTracks.size == 0:                              # break when no more incoming tracks
  			break
  		track = np.argmin(abs(pfreqt[i] - tfreq[incomingTracks]))   # closest incoming track to peak
  		freqDistance = abs(pfreq[i] - tfreq[incomingTracks[track]]) # measure freq distance
  		if freqDistance < (freqDevOffset + freqDevSlope * pfreq[i]):  # choose track if distance is small 
  		    newTracks[incomingTracks[track]] = i                      # assign peak index to track index
  		    incomingTracks = np.delete(incomingTracks, track)         # delete index of track in incomming tracks
  indext = np.array(np.nonzero(newTracks != -1), dtype=np.int)[0]   # indexes of assigned tracks
  if indext.size > 0:
    indexp = newTracks[indext]                                    # indexes of assigned peaks
    tfreqn[indext] = pfreqt[indexp]                               # output freq tracks 
    tmagn[indext] = pmagt[indexp]                                 # output mag tracks 
    tphasen[indext] = pphaset[indexp]                             # output phase tracks 
    pfreqt= np.delete(pfreqt, indexp)                             # delete used peaks
    pmagt= np.delete(pmagt, indexp)                               # delete used peaks
    pphaset= np.delete(pphaset, indexp)                           # delete used peaks

  # create new tracks from non used peaks
  emptyt = np.array(np.nonzero(tfreq == 0), dtype=np.int)[0]      # indexes of empty incoming tracks
  peaksleft = np.argsort(-pmagt)                                  # sort left peaks by magnitude
  if ((peaksleft.size > 0) & (emptyt.size >= peaksleft.size)):    # fill empty tracks
  		tfreqn[emptyt[:peaksleft.size]] = pfreqt[peaksleft]
  		tmagn[emptyt[:peaksleft.size]] = pmagt[peaksleft]
  		tphasen[emptyt[:peaksleft.size]] = pphaset[peaksleft]
  elif ((peaksleft.size > 0) & (emptyt.size < peaksleft.size)):   # add more tracks if necessary
  		tfreqn[emptyt] = pfreqt[peaksleft[:emptyt.size]]
  		tmagn[emptyt] = pmagt[peaksleft[:emptyt.size]]
  		tphasen[emptyt] = pphaset[peaksleft[:emptyt.size]]
  		tfreqn = np.append(tfreqn, pfreqt[peaksleft[emptyt.size:]])
  		tmagn = np.append(tmagn, pmagt[peaksleft[emptyt.size:]])
  		tphasen = np.append(tphasen, pphaset[peaksleft[emptyt.size:]])
  
  return tfreqn, tmagn, tphasen