import numpy as np, scipy.signal as sg, scipy.io.wavfile as wf, sys, json, os
INFO=json.loads(sys.argv[1]); OUTD=sys.argv[2]
SR=44100; T=INFO['TEND']; N=INFO['N']; TB=INFO['TB']; S0=INFO['S0']; S1=INFO['S1']; TALL=INFO['TALL']; n=int(SR*T); t=np.arange(n)/SR
rng=np.random.RandomState(12)
def noise(d): return rng.randn(int(d*SR))
def bp(x,lo,hi): b,a=sg.butter(2,[lo/(SR/2),hi/(SR/2)],'band'); return sg.lfilter(b,a,x)
def lp(x,f): b,a=sg.butter(2,f/(SR/2)); return sg.lfilter(b,a,x)
class Bus:
    def __init__(s): s.L=np.zeros(n); s.R=np.zeros(n)
    def add(s,sig,t0,pan=0.0,g=1.0):
        i=int(t0*SR)
        if i<0 or i>=n: return
        m=min(len(sig),n-i)
        s.L[i:i+m]+=sig[:m]*g*(1-max(pan,0)); s.R[i:i+m]+=sig[:m]*g*(1+min(pan,0))
SFX=Bus(); LEAD=Bus(); PAD=Bus(); DRUM=Bus(); BASS=Bus()
def adsr(d,a,r):
    k=np.arange(int(d*SR))/SR
    return np.minimum(k/a,1)*np.clip((d-k)/r,0,1)
def piano2(f,vel=1.0,d=2.4):
    k=np.arange(int(d*SR))/SR; dec=0.9+f/500.0; x=0
    for h,a in ((1,1.0),(2,0.55),(3,0.28),(4,0.16),(5,0.08),(6,0.05)):
        x=x+a*np.sin(2*np.pi*f*h*(1+0.0004*h*h)*k)*np.exp(-k*dec*(1+0.35*(h-1)))
    ham=lp(noise(0.03),5000)*np.exp(-np.arange(int(0.03*SR))/SR*120)*0.25
    x[:len(ham)]+=ham
    return lp(x,5200)*np.minimum(k/0.003,1)*vel
def strings2(f,d,att=0.5,rel=0.9):
    k=np.arange(int(d*SR))/SR
    vib=1+0.003*np.sin(2*np.pi*5.1*k)*np.minimum(k/0.9,1)
    ph=2*np.pi*f*np.cumsum(vib)/SR; x=0
    for c in (-9,0,9): x=x+sg.sawtooth(ph*2**(c/1200))
    return (lp(x/3,2600)+0.25*np.sin(ph*0.5))*adsr(d,att,rel)
def harp(f,d=1.6):
    k=np.arange(int(d*SR))/SR
    return (np.sin(2*np.pi*f*k)+0.25*np.sin(2*np.pi*2*f*k))*np.exp(-k*3.0)*np.minimum(k/0.002,1)
def softkick():
    k=np.arange(int(0.45*SR))/SR
    return np.sin(2*np.pi*np.cumsum(42+70*np.exp(-k*30))/SR)*np.exp(-k*9)
def brush(op=0.09):
    k=np.arange(int(op*SR))/SR; b_,a_=sg.butter(2,5500/(SR/2),'high'); return sg.lfilter(b_,a_,noise(op))*np.exp(-k*38)
def timp(f=90.0,d=1.2):
    k=np.arange(int(d*SR))/SR
    return np.sin(2*np.pi*np.cumsum(f*(1+0.5*np.exp(-k*25)))/SR)*np.exp(-k*3.0)+0.25*lp(noise(d),400)*np.exp(-k*12)
def sub(f,d):
    k=np.arange(int(d*SR))/SR
    return np.tanh(np.sin(2*np.pi*np.cumsum(f*(1+1.0*np.exp(-k*60)))/SR)*np.exp(-k*1.6)*2.0)*0.6
def crash(d=3.0):
    k=np.arange(int(d*SR))/SR; b_,a_=sg.butter(2,3500/(SR/2),'high'); return sg.lfilter(b_,a_,noise(d))*np.exp(-k*1.6)
def whoosh(bus,t_end,d=0.5,lo=400,hi=3500,up=True,g=0.5,pan=0.0):
    k=np.arange(int(d*SR))/SR; nz=noise(d); out=np.zeros_like(nz)
    for j in range(0,len(nz),1024):
        fr=j/len(nz); f=(lo+(hi-lo)*fr) if up else (hi-(hi-lo)*fr)
        out[j:j+1024]=bp(nz[j:j+1024],f*0.6,f*1.5)
    bus.add(out*np.sin(np.pi*np.clip(k/d,0,1))**1.5*g,t_end-d,pan=pan)
def chime(bus,tt,f,g=0.12,pan=0.0):
    k=np.arange(int(1.6*SR))/SR
    bus.add((np.sin(2*np.pi*f*k)+0.3*np.sin(2*np.pi*2*f*k)+0.1*np.sin(2*np.pi*3.01*f*k))*np.exp(-k*3.0)*np.minimum(k/0.002,1),tt,pan=pan,g=g)
def shimmer(bus,tt,d=1.5,g=0.3,pan=0.2):
    kk=np.arange(int(d*SR))/SR
    bus.add(bp(noise(d),3000,10000)*np.sin(np.pi*np.clip(kk/d,0,1))**2,tt,pan=pan,g=g)
def riser(d,lo,hi,pw=1.5):
    k=np.arange(int(d*SR))/SR; nz=noise(d); sw=np.zeros_like(nz)
    for j in range(0,len(nz),2048):
        f=lo+(hi-lo)*(j/len(nz))**pw; sw[j:j+2048]=bp(nz[j:j+2048],f*0.6,f*1.4)
    return sw*(k/d)**1.6
PENT=[1396.91,1567.98,1760.0,2093.0,2349.3,2793.0]
# ================= SFX: spinning-coin entrance =================
PENT=[1396.91,1567.98,1760.0,2093.0,2349.3,2793.0]
kk0=np.arange(int(2.4*SR))/SR
SFX.add(np.sin(2*np.pi*(40+40*np.exp(-kk0*6))*kk0)*np.exp(-kk0*2.0),0.85,g=0.8)      # coin enters
SFX.add(lp(noise(2.4),600)*np.exp(-kk0*3),0.85,g=0.4); shimmer(SFX,0.85,1.6,0.3,0.0)
SFX.add(riser(3.0,500,8000,1.4),1.2,g=0.45)
for h in range(1,7):                                   # half-turn swishes (spin decelerates)
    x_=(0.5*h)/3.0; u_=1-(1-x_)**(1/3.0) if x_<1 else 1.0
    tt=0.8+3.4*u_
    whoosh(SFX,tt,0.45,500+h*250,2500+h*600,True,0.45,pan=(-1)**h*0.5)
    chime(SFX,tt,[1396.91,1567.98,1760.0,2093.0,2349.3,2793.0][h-1],0.07,pan=(-1)**h*0.4)
kk2=np.arange(int(2.6*SR))/SR
SFX.add(np.sin(2*np.pi*(40+34*np.exp(-kk2*7))*kk2)*np.exp(-kk2*2.2),4.2,g=0.9)     # settle impact
SFX.add(lp(noise(2.6),700)*np.exp(-kk2*3),4.2,g=0.5); shimmer(SFX,4.2,2.4,0.4,0.0)
for j,f in enumerate((1396.91,1760.0,2093.0,2637.0)): chime(SFX,4.22+0.1*j,f,0.12,pan=-0.3+0.2*j)
chime(SFX,3.2,1760.0,0.10); chime(SFX,3.5,2349.3,0.10)
whoosh(SFX,6.4,1.4,400,3500,True,0.5)
chime(SFX,6.4,2349.3,0.12); shimmer(SFX,6.4,1.4,0.25,-0.3)
chime(SFX,5.9,1760.0,0.10); chime(SFX,6.3,2093.0,0.10); chime(SFX,6.7,2349.3,0.10)
# ---- members: pop + chime + flight whoosh + landing tick
for i in range(N):
    ti=6.9+1.0*i
    kk=np.arange(int(0.5*SR))/SR
    SFX.add(np.sin(2*np.pi*(90+60*np.exp(-kk*12))*kk)*np.exp(-kk*8),ti,g=0.4)
    chime(SFX,ti+0.05,PENT[i%6]*(1.0 if i<6 else 0.5),0.14,pan=0.3*(i-4)/4)
    whoosh(SFX,ti+1.0,0.45,1200,5000,True,0.22)
    d_=0.05; k2=np.arange(int(d_*SR))/SR
    SFX.add(bp(noise(d_),1800,6500)*np.exp(-k2*80),ti+1.0,g=0.3)
# ---- banner reveal at 16.7 (mosaic)
TBv=TB
SFX.add(riser(1.0,1500,9000),TBv-0.9,g=0.5)
whoosh(SFX,TBv+0.3,1.0,300,3800,True,0.4)
kk3=np.arange(int(2.4*SR))/SR
SFX.add(np.sin(2*np.pi*(46+34*np.exp(-kk3*8))*kk3)*np.exp(-kk3*2.6),TBv,g=0.85)
shimmer(SFX,TBv,2.0,0.35,0.0)
for j,f in enumerate((1396.91,1760.0,2093.0,2793.0)): chime(SFX,TBv+0.1+0.12*j,f,0.12,pan=-0.3+0.2*j)
for j in range(14): chime(SFX,TBv+0.2+0.1*j,PENT[j%6]*1.5,0.05,pan=-0.5+0.08*j)
for g0 in [g for g in ([S0,S1] if TALL else [TB+4.8,TB+10.3]) if g<T-1.6]: shimmer(SFX,g0,1.5,0.25); whoosh(SFX,g0+0.5,0.9,500,3000,True,0.15)
# ================= MUSIC =================
for fq,gg in ((87.31,0.8),(174.61,1.0),(261.63,0.9),(440.0,0.6),(523.25,0.5)):
    PAD.add(strings2(fq,3.6,2.8,0.4),0.7,g=0.10*gg)
d=3.3; k=np.arange(int(d*SR))/SR
ph=2*np.pi*np.cumsum(174.61*(4**(k/d)))/SR
LEAD.add(lp(np.sin(ph)*(k/d)**2*0.12,3000),0.9)
for j in range(10): DRUM.add(timp(70.0,0.6),1.6+2.5*(1-(1-j/10)**1.6),g=0.06+0.06*j/10)
DRUM.add(timp(60.0,2.0),4.2,g=0.9); BASS.add(sub(43.65,2.4),4.2,g=0.9); DRUM.add(crash(3.2),4.2,g=0.4)
for fq in (174.61,220.0,261.63,349.23,440.0,523.25): PAD.add(strings2(fq,2.4,0.05,1.2),4.2,g=0.09)
for fq in (174.61,261.63,329.63,440.0): LEAD.add(piano2(fq,1.0,3.0),4.2,g=0.25)
for j,fq in enumerate((261.63,329.63,392.0,523.25,659.25,783.99)): LEAD.add(harp(fq),5.0+0.07*j,pan=-0.2+0.08*j,g=0.16)
CHD={'Fmaj7':dict(r=87.31,ch=(174.61,220.0,261.63,329.63),ar=[174.61,261.63,329.63,440.0,329.63,261.63,220.0,261.63]),
     'Am7':dict(r=110.0,ch=(164.81,196.0,261.63,329.63),ar=[164.81,261.63,329.63,392.0,329.63,261.63,196.0,261.63]),
     'Bbmaj7':dict(r=116.54,ch=(174.61,220.0,293.66,349.23),ar=[174.61,293.66,349.23,440.0,349.23,293.66,220.0,293.66]),
     'C':dict(r=130.81,ch=(196.0,261.63,329.63,392.0),ar=[196.0,329.63,392.0,523.25,392.0,329.63,261.63,329.63]),
     'Dm7':dict(r=146.83,ch=(174.61,220.0,261.63,349.23),ar=[174.61,261.63,349.23,440.0,349.23,261.63,220.0,261.63]),
     'Gm7':dict(r=98.0,ch=(196.0,233.08,293.66,349.23),ar=[196.0,293.66,349.23,466.16,349.23,293.66,233.08,293.66])}
SEQ=['Fmaj7','Am7','Bbmaj7','C','Dm7','Bbmaj7','Gm7','C']
MELS=[[(0,1.5,523.25),(1.5,0.5,587.33),(2,2,659.25)],
      [(0,1.5,659.25),(1.5,0.5,587.33),(2,1,523.25),(3,1,440.0)],
      [(0,1,587.33),(1,1,698.46),(2,2,880.0)],
      [(0,1.5,783.99),(1.5,0.5,698.46),(2,2,659.25)],
      [(0,1.5,698.46),(1.5,0.5,659.25),(2,1,587.33),(3,1,523.25)],
      [(0,1,698.46),(1,1,587.33),(2,2,466.16)],
      [(0,1.5,466.16),(1.5,0.5,523.25),(2,2,587.33)],
      [(0,2,659.25),(2,2,783.99)]]
B=0.46875
def bar(t0,name,level=1.0,arp=True,mel=None,mel_g=1.0,pad_g=0.05,oct2=False):
    c=CHD[name]
    for fq in c['ch']: PAD.add(strings2(fq,2.7),t0,g=pad_g*level)
    for b in (0,2):
        kb=np.arange(int(1.4*SR))/SR
        BASS.add((np.sin(2*np.pi*c['r']*kb)+0.3*np.sin(2*np.pi*2*c['r']*kb))*np.exp(-kb*1.4)*np.minimum(kb/0.01,1),t0+b*B,g=0.32*level)
    if arp:
        vels=[0.9,0.5,0.6,0.75,0.8,0.5,0.55,0.5]
        for sx in range(8): LEAD.add(piano2(c['ar'][sx],vels[sx]*level,1.8),t0+sx*B/2,pan=0.25*(-1)**sx,g=0.17)
    if mel:
        for (bo,bd,fq) in mel:
            LEAD.add(piano2(fq,1.0,2.4),t0+bo*B,g=0.26*mel_g); PAD.add(strings2(fq,bd*B+0.5,0.18,0.5),t0+bo*B,g=0.06*mel_g)
            if oct2: LEAD.add(piano2(fq*2,0.8,1.6),t0+bo*B+0.02,g=0.08*mel_g)

# bars: 1.875 s each; downbeats at 4.2 (settle) and 11.7 (banner reveal)
MM={'Fmaj7':0,'Am7':1,'Bbmaj7':2,'C':3,'Dm7':4,'Gm7':6}
def B_(t0,nm,lvl=1.0,mg=1.0,pad_g=0.05,oct2=False,arp=True,mel=True):
    bar(t0,nm,lvl,arp=arp,mel=MELS[MM[nm]] if mel else None,mel_g=mg,pad_g=pad_g,oct2=oct2)
BL=1.875
SQ1=['Fmaj7','Bbmaj7','C','Gm7','Am7','Dm7','Bbmaj7','C']
n1=max(1,int(round((TB-1.25-4.2)/BL)))
for k_ in range(n1): B_(4.2+k_*BL,SQ1[k_%8],0.7 if k_==0 else min(1.0,0.8+0.05*k_),0.8 if k_==0 else 1.0,arp=(k_>0))
for fq in (196.0,261.63,329.63,392.0,523.25): PAD.add(strings2(fq,1.6,1.0,0.1),TB-1.6,g=0.09)
DRUM.add(timp(60.0,2.2),TB,g=0.8); BASS.add(sub(43.65,2.4),TB,g=0.6)
for j,fq in enumerate((261.63,329.63,392.0,523.25,659.25,783.99,1046.5)): LEAD.add(harp(fq,2.0),TB+0.05+0.08*j,pan=-0.3+0.1*j,g=0.14)
SQ=['Fmaj7','Bbmaj7','Am7','C','Dm7','Bbmaj7','Gm7','C']
n2=max(1,int(round((T-1.7-TB)/BL)))
for k_ in range(n2): B_(TB+k_*BL,SQ[k_%8],1.05 if k_%2==0 else 1.0,1.1,pad_g=0.07,oct2=(k_<4))
tf=TB+n2*BL
for fq in (174.61,220.0,261.63,349.23,440.0): PAD.add(strings2(fq,2.4,0.8,1.3),tf,g=0.09)
for fq in (174.61,261.63,329.63,440.0,523.25): LEAD.add(piano2(fq,0.9,3.0),tf,g=0.20)
LEAD.add(piano2(659.25,0.9,2.6),tf+0.1,g=0.22)
BASS.add(sub(43.65,2.0),tf,g=0.5)
for j,fq in enumerate((523.25,659.25,783.99,1046.5,1318.5)): LEAD.add(harp(fq,2.0),tf+0.1*j,pan=-0.3+0.15*j,g=0.12)
# ================= mix =================
def reverb(x,sec=3.6,seed=1):
    r=np.random.RandomState(seed); m=int(sec*SR)
    ir=r.randn(m)*np.exp(-np.arange(m)/SR*2.8); ir[:int(0.012*SR)]*=0; ir=lp(ir,6500)
    return sg.fftconvolve(x,ir)[:len(x)]/np.abs(ir).sum()*40
fade=np.clip((T-t)/1.4,0,1)
mus=[None,None]
for ch,seed in ((0,1),(1,2)):
    g=lambda b:(b.L if ch==0 else b.R)
    dry=g(LEAD)+g(PAD)+g(DRUM)+g(BASS)
    wet=reverb(g(LEAD)*0.6+g(PAD)*0.5,seed=seed)
    mus[ch]=(dry+wet*0.6)*fade
pk=max(np.abs(mus[0]).max(),np.abs(mus[1]).max())
mL=mus[0]/pk*0.62; mR=mus[1]/pk*0.62
def fin(a,b):
    m=max(np.abs(a).max(),np.abs(b).max()); return a/m,b/m
L1,R1=fin(SFX.L*0.6+mL,SFX.R*0.6+mR)
L2,R2=fin(SFX.L,SFX.R)
wf.write(os.path.join(OUTD,'music.wav'),SR,(np.stack([L1,R1],1)*0.7*32767).astype(np.int16))
wf.write(os.path.join(OUTD,'sfx.wav'),SR,(np.stack([L2,R2],1)*0.7*32767).astype(np.int16))
print('ok')
