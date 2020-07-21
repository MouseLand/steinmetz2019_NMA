import numpy as np
import os, csv
from numba import jit
from scipy.sparse import csr_matrix
from scipy.interpolate import interp1d
from sklearn.decomposition import TruncatedSVD

def get_good_cells(fdirpath):
    # location in brain of each neuron
    brain_loc = os.path.join(fdirpath, "channels.brainLocation.tsv")

    good_cells = (np.load(os.path.join(fdirpath, "clusters._phy_annotation.npy")) >= 2 ).flatten()
    clust_channel = np.load(os.path.join(fdirpath, "clusters.peakChannel.npy")).astype(int) - 1
    br = []
    with open(brain_loc, 'r') as tsv:
        tsvin = csv.reader(tsv, delimiter="\t")
        k=0
        for row in tsvin:
            if k>0:
                br.append(row[-1])
            k+=1
    br = np.array(br)
    good_cells = np.logical_and(good_cells, clust_channel.flatten()<len(br))
    brain_region = br[clust_channel[:,0]]


    return good_cells, brain_region, br


def get_waves(fdirpath):
    fname = os.path.join(fdirpath, "clusters.waveformDuration.npy")
    twav = np.load(fname)

    fname = os.path.join(fdirpath, "clusters.templateWaveforms.npy")
    W = np.load(fname)

    fname = os.path.join(fdirpath, "clusters.templateWaveformChans.npy")
    ichan = np.load(fname).astype('int32')

    u = np.zeros((W.shape[0], 3, 384))
    w = np.zeros((W.shape[0], 82, 3))

    for j in range(W.shape[0]):
        wU  = TruncatedSVD(n_components = 3).fit(W[j]).components_
        wW = W[j] @ wU.T
        u[j][:, ichan[j]%384] = wU
        w[j] = wW

    return twav, w, u


def get_probe(fdirpath, br):
    prb_name = os.path.join(fdirpath, "probes.rawFilename.tsv")
    prb = []
    with open(prb_name, 'r') as tsv:
        tsvin = csv.reader(tsv, delimiter="\t")
        for row in tsvin:
            prb.append(row[-1])
        prb = prb[1:]
    for ip in range(len(prb)):
        pparts = prb[ip].split('_')
        prb[ip] = '%s_%s_%s_%s'%(pparts[0], pparts[1], pparts[2], pparts[3])

    brow = []
    blfp = []
    for iprobe in range(len(prb)):
        ch_prb = np.load(os.path.join(fdirpath, "channels.probe.npy")).astype(int)
        raw_row = np.load(os.path.join(fdirpath, "channels.rawRow.npy")).astype(int)
        ich = (ch_prb.flatten()==iprobe).nonzero()[0]
        bunq = np.unique(br[ich])
        bunq = bunq[bunq!='root']
        nareas = len(bunq)
        brow.append([])
        for j in range(nareas):
            bid = br[ich]==bunq[j]
            brow[-1].append(raw_row[ich[bid], 0])
        blfp.append(bunq)
    return prb, blfp, brow


def get_LFP(fdirpath, br, etime, dT, dt, T0):
    prb, blfp, brow = get_probe(fdirpath, br)
    bsize = 100000
    nbytesread = 385 * bsize * 2

    L = []
    BA_LFP = []
    for ip in range(len(prb)):
        BA_LFP.extend(blfp[ip])

        root = 'G:/LFP'
        fname_lfp = '%s_t0.imec.lf.bin'%(prb[ip])

        LFP = []
        with open(os.path.join(root, fname_lfp), 'rb') as lfp_file:
            while True:
                buff = lfp_file.read(nbytesread)
                data = np.frombuffer(buff, dtype=np.int16, offset=0)
                if data.size==0:
                    break
                data = np.reshape(data, (-1, 385))

                nareas = len(brow[ip])
                lfp = np.zeros((data.shape[0], nareas))
                for j in range(nareas):
                    lfp[:,j] = data[:, brow[ip][j]].mean(-1)
                LFP.extend(lfp)
        LFP = np.array(LFP)
        fname_lfp_times = '%s_t0.imec.lf.timestamps.npy'%(prb[ip])
        lfp_times = np.load(os.path.join(fdirpath, fname_lfp_times))
        L.extend(ppsth(LFP, lfp_times,  etime, dT, dt))

    L = np.array(L)
    L = L - np.expand_dims(np.mean(L[:,:,:int(T0//dt)], axis=-1), axis=-1)

    return L, BA_LFP

def get_passive(fdirpath):
    vis_right_p = np.load(os.path.join(fdirpath, "passiveVisual.contrastRight.npy")).flatten()
    vis_left_p = np.load(os.path.join(fdirpath, "passiveVisual.contrastLeft.npy")).flatten()
    vis_times_p = np.load(os.path.join(fdirpath,   "passiveVisual.times.npy"))
    return vis_times_p, vis_right_p, vis_left_p


def get_event_types(fdirpath):
    response = np.load(os.path.join(fdirpath, "trials.response_choice.npy")).flatten()
    vis_right = np.load(os.path.join(fdirpath, "trials.visualStim_contrastRight.npy")).flatten()
    vis_left = np.load(os.path.join(fdirpath, "trials.visualStim_contrastLeft.npy")).flatten()
    feedback_type = np.load(os.path.join(fdirpath, "trials.feedbackType.npy")).flatten()

    return response, vis_right, vis_left, feedback_type

def get_event_times(fdirpath):
    response_times = np.load(os.path.join(fdirpath, "trials.response_times.npy"))
    visual_times = np.load(os.path.join(fdirpath,   "trials.visualStim_times.npy"))
    gocue = np.load(os.path.join(fdirpath,   "trials.goCue_times.npy"))
    feedback = np.load(os.path.join(fdirpath,   "trials.feedback_times.npy"))

    rsp = response_times - visual_times
    feedback = feedback - visual_times
    gocue = gocue - visual_times

    return response_times, visual_times, rsp, gocue, feedback

def get_wheel(fdirpath):
    wheel = np.load(os.path.join(fdirpath, "wheel.position.npy")).flatten()
    wheel_times = np.load(os.path.join(fdirpath,   "wheel.timestamps.npy"))
    return wheel, wheel_times

def get_pup(fdirpath):
    pup = np.load(os.path.join(fdirpath, "eye.area.npy"))
    pup_times = np.load(os.path.join(fdirpath,  "eye.timestamps.npy"))
    xy = np.load(os.path.join(fdirpath, "eye.xyPos.npy"))

    return pup, xy, pup_times

def get_spikes(fdirpath):
    stimes = np.load(os.path.join(fdirpath, "spikes.times.npy")).flatten()
    sclust = np.load(os.path.join(fdirpath, "spikes.clusters.npy")).flatten()
    return stimes, sclust

def first_spikes(stimes, t0):
    tlow = 0
    thigh = len(stimes)

    while thigh>tlow+1:
        thalf = (thigh + tlow)//2
        sthalf = stimes[thalf]
        if t0 >= sthalf:
            tlow = thalf
        else:
            thigh = thalf
    return thigh

def wpsth(wheel, wheel_times, etime, dT, dt):
    ntrials = len(etime)
    NT = int(dT/dt)
    f = interp1d(wheel_times[:,1], wheel_times[:,0], fill_value='extrapolate')
    S  = np.zeros((ntrials, NT))
    for j in range(ntrials):
        tsamp = f(np.arange(etime[j], etime[j]+dT+1e-5, dt)).astype('int32')
        S[j,:] = wheel[tsamp[1:]] - wheel[tsamp[:-1]]
    return S

def ppsth(pup, pup_times, etime, dT, dt):
    nk = pup.shape[-1]
    ntrials = len(etime)
    NT = int(dT/dt)
    f = interp1d(pup_times[:,1], pup_times[:,0], fill_value='extrapolate')
    S  = np.zeros((nk, ntrials, NT))
    for k in range(nk):
        for j in range(ntrials):
            tsamp = f(np.arange(etime[j], etime[j]+dT-1e-5, dt) + dt/2).astype('int32')
            S[k, j,:] = pup[tsamp, k]
    return S

def psth(stimes, sclust, etime, dT, dt):
    NN = np.max(sclust)+1
    NT = int(dT/dt)
    ntrials = len(etime)

    S  = np.zeros((NN, ntrials, NT))
    for j in range(ntrials):
        k1   = first_spikes(stimes, etime[j])
        k2   = first_spikes(stimes, etime[j]+dT)
        st   = stimes[k1:k2] - etime[j]
        clu  = sclust[k1:k2]
        S[:,j,:] = csr_matrix((np.ones(k2-k1), (clu, np.int32(st/dt))), shape=(NN,NT)).todense()

    return S
