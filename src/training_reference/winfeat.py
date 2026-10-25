"""Window features (skeleton+IMU) for every unit; windows of WIN seconds, stride STRIDE seconds."""
import numpy as np, glob, os, sys, time
from multiprocessing import Pool
sys.path.insert(0,'/PARTICIPANT_WORKSPACE/historical_seq_workspace/cuhk_large_seq')
from skel_feat import feat_from_kp, FDIM
RAW='/PARTICIPANT_WORKSPACE/cuhk_large_takeover/attack/raw'
WIN=float(os.environ.get('WIN',2.0)); STRIDE=float(os.environ.get('STRIDE',1.0)); FPS=10.0
def imu_feat(x):  # x (n,5,12) may contain nan
    out=[]
    for k in range(5):
        v=x[:,k,:]
        if len(v)==0 or np.isnan(v).all():
            out.append(np.zeros(12*4+3,np.float32)); out.append(np.array([1.0],np.float32)); continue
        v=np.nan_to_num(v)
        acc=np.linalg.norm(v[:,:3],axis=1); gyr=np.linalg.norm(v[:,3:6],axis=1)
        out.append(np.concatenate([v.mean(0),v.std(0),v.min(0),v.max(0),[acc.std(),gyr.mean(),gyr.std()]]).astype(np.float32))
        out.append(np.array([0.0],np.float32))
    return np.concatenate(out)
IDIM=5*(12*4+3+1)
def unit_windows(f):
    d=np.load(f); sk=d['sk']; fn=d['fn']; imu=d['imu']; imu_t=d['imu_t']
    if len(sk)<3: return None
    # skeleton time in seconds relative to first frame (frame numbers, 10fps)
    t_sk=(fn-fn.min())/FPS; T=t_sk.max()+0.1
    # imu time: assume same span -> rescale imu grid to [0,T]
    if len(imu_t)>1: t_imu=imu_t*(T/max(imu_t.max(),1e-3))
    else: t_imu=None
    starts=np.arange(0,max(T-WIN,0)+1e-6,STRIDE)
    if len(starts)==0: starts=np.array([0.0])
    feats=[];cent=[]
    for s in starts:
        m=(t_sk>=s)&(t_sk<s+WIN)
        kp=sk[m]
        if len(kp)<3:
            fs=np.zeros(FDIM,np.float32)
        else:
            fs=np.asarray(feat_from_kp(kp),np.float32)
            if fs.shape[0]!=FDIM: fs=np.resize(fs,FDIM)
        if t_imu is not None:
            mi=(t_imu>=s)&(t_imu<s+WIN); fi=imu_feat(imu[mi])
        else: fi=imu_feat(np.zeros((0,5,12),np.float32))
        feats.append(np.concatenate([fs,fi])); cent.append((s+WIN/2)/T)
    return np.stack(feats).astype(np.float32), np.array(cent,np.float32), T
def work(f):
    u=os.path.basename(f)[:-4].replace('__','/')
    r=unit_windows(f)
    return u,r
if __name__=='__main__':
    fs=sorted(glob.glob(f'{RAW}/*.npz'))
    t0=time.time(); X={};C={};Tt={}
    with Pool(7) as p:
        for i,(u,r) in enumerate(p.imap_unordered(work,fs,chunksize=16)):
            if r is None: continue
            X[u],C[u],Tt[u]=r
    np.savez_compressed(f'win_W{WIN}_S{STRIDE}.npz',units=np.array(list(X)),X=np.array([X[u] for u in X],dtype=object),C=np.array([C[u] for u in X],dtype=object),T=np.array([Tt[u] for u in X]))
    print('done',len(X),f'{time.time()-t0:.0f}s', 'dim', next(iter(X.values())).shape)
