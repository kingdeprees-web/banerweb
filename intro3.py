import cv2, numpy as np, math, sys, subprocess, os
from PIL import Image, ImageDraw, ImageFont
W,H,FPS=1080,1920,30
TEND=13.0
rng=np.random.RandomState(11)
def sstep(x): x=np.clip(x,0,1); return x*x*(3-2*x)
def eo3(x): x=np.clip(x,0,1); return 1-(1-x)**3
def eio3(x): x=np.clip(x,0,1); return 4*x**3 if x<.5 else 1-(-2*x+2)**3/2
def eisine(x): x=np.clip(x,0,1); return -(math.cos(math.pi*x)-1)/2
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
GOLD=np.array([255,205,120],np.float32)
FB='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

# ---------------- coin sprite
src=Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'assets','logo_source.webp')).convert('RGB')
cx0,cy0,R0=311.5,308.5,292.5
SD=int(2*R0)+6
x0=int(round(cx0-SD/2)); y0=int(round(cy0-SD/2))
coin=src.crop((x0,y0,x0+SD,y0+SD)).convert('RGBA')
m=Image.new('L',(SD*4,SD*4),0)
ImageDraw.Draw(m).ellipse(((SD/2-R0)*4,(SD/2-R0)*4,(SD/2+R0)*4,(SD/2+R0)*4),fill=255)
coin.putalpha(m.resize((SD,SD),Image.LANCZOS))
coinA=np.array(coin).astype(np.float32); coinA[...,3]/=255.0
gy,gx=np.mgrid[0:SD,0:SD].astype(np.float32)
shine_u=(gx+0.55*gy)/(SD*1.55)
coin_mask=np.where(coinA[...,3]>0,1,0).astype(np.float32)

# ---------------- text
def render_text(txt,size,top,bot):
    font=ImageFont.truetype(FB,size,layout_engine=ImageFont.Layout.RAQM)
    wlen=int(font.getlength(txt,direction='rtl',language='fa'))+size
    hh=int(size*2.2)
    im=Image.new('L',(wlen,hh),0)
    ImageDraw.Draw(im).text((wlen//2,hh//2),txt,font=font,fill=255,anchor='mm',direction='rtl',language='fa')
    a=np.array(im).astype(np.float32)/255
    ys,xs=np.where(a>0.02)
    pad=8
    a=a[max(0,ys.min()-pad):ys.max()+pad, max(0,xs.min()-pad):xs.max()+pad]
    h,w=a.shape
    g=np.linspace(0,1,h)[:,None,None]
    rgb=(np.array(top,np.float32)*(1-g)+np.array(bot,np.float32)*g)*np.ones((1,w,1),np.float32)
    return rgb,a
TXT={}
def prep(name,txt,size,top,bot,col):
    rgb,al=render_text(txt,size,top,bot)
    rgbb=cv2.GaussianBlur(rgb*al[...,None],(0,0),7); alb=cv2.GaussianBlur(al,(0,0),7)
    PADG=80
    glowp=cv2.GaussianBlur(np.pad(al,PADG),(0,0),26)
    TXT[name]=dict(rgb=rgb,al=al,rgbb=rgbb,alb=alb,glowp=glowp,col=np.array(col,np.float32))
GT,GB=(255,232,160),(224,162,58)
WT,WB=(255,255,255),(208,222,238)
prep('label','دوره تخصصی مقدماتی',64,WT,WB,(180,210,255))
prep('t1','بهسازی لرزه‌ای',126,GT,GB,(255,200,100))
prep('t2','سازه',126,GT,GB,(255,200,100))
prep('with','با تدریس',50,WT,WB,(180,210,255))
prep('dr','جناب دکتر فریدونی',88,GT,GB,(255,200,100))
prep('org1','توسط انجمن سازندگان ذی‌صلاح',50,WT,WB,(180,210,255))
prep('org2','ساختمان شهرستان ارومیه',50,WT,WB,(180,210,255))
prep('a1','انجمن سازندگان ذی‌صلاح',60,WT,WB,(180,210,255))
prep('a2','ساختمان شهرستان ارومیه',60,WT,WB,(180,210,255))

def draw_text(cv,name,t,t_in,t_out,cx,cy,dur=1.1,fade_out=0.6):
    T=TXT[name]
    env=sstep((t-t_in)/0.25)*(1-sstep((t-t_out)/fade_out))
    if env<=0.003: return
    p=eo3((t-t_in)/dur)
    h,w=T['al'].shape
    xs=np.arange(w,dtype=np.float32)
    front=(1-p)*(w+110)-110
    mloc=sstep((xs-front)/110)[None,:]
    rise=int(16*(1-eo3((t-t_in)/1.4)))-int(10*sstep((t-t_out)/fade_out))
    x=int(cx-w/2); y=int(cy-h/2)+rise
    ya,yb=max(0,y),min(H,y+h); xa,xb=max(0,x),min(W,x+w)
    if ya>=yb or xa>=xb: return
    sl=(slice(ya-y,yb-y),slice(xa-x,xb-x))
    m_=mloc[:,sl[1]]
    al=T['al'][sl]; rgb=T['rgb'][sl]; alb=T['alb'][sl]; rgbb=T['rgbb'][sl]
    edge=np.clip(1-np.abs(m_-0.5)*2,0,1)
    k=edge*0.9
    a_use=(al*(1-k)+alb*k)*m_*env
    rgb_use=rgb*(1-k[...,None])+(rgbb/np.maximum(alb[...,None],1e-3))*k[...,None]
    xs2=xs[sl[1]][None,:]
    band=np.exp(-((xs2-front-55)/38.0)**2)*(p<0.999)
    rgb_use=np.clip(rgb_use+band[...,None]*np.array([255,235,180],np.float32)*0.7,0,255)
    reg=cv[ya:yb,xa:xb]
    reg[:]=reg*(1-a_use[...,None])+rgb_use*a_use[...,None]
    G=T['glowp']; gh,gw=G.shape; gx_=x-80; gy_=y-80
    ya2,yb2=max(0,gy_),min(H,gy_+gh); xa2,xb2=max(0,gx_),min(W,gx_+gw)
    if ya2<yb2 and xa2<xb2:
        gsl=G[ya2-gy_:yb2-gy_, xa2-gx_:xb2-gx_]
        xs_g=(np.arange(xa2,xb2)-x).astype(np.float32)[None,:]
        mg=sstep((xs_g-front)/110)
        cv[ya2:yb2,xa2:xb2]+=(gsl*mg)[...,None]*T['col']*(0.55*env)

def draw_line(cv,t,t_in,t_out,cy,halfw):
    env=sstep((t-t_in)/0.2)*(1-sstep((t-t_out)/0.5))
    if env<=0.003: return
    p=eo3((t-t_in)/1.2); hw_=int(halfw*p)
    if hw_>4:
        y0_=max(0,cy-16); y1_=min(H,cy+16)
        ov=np.zeros((y1_-y0_,W,3),np.float32); c=cy-y0_
        cv2.line(ov,(540-hw_,c),(540-16,c),(235,190,95),2,cv2.LINE_AA)
        cv2.line(ov,(540+16,c),(540+hw_,c),(235,190,95),2,cv2.LINE_AA)
        pts=np.array([[540,c-10],[550,c],[540,c+10],[530,c]],np.int32)
        cv2.fillPoly(ov,[pts],(245,205,120),cv2.LINE_AA)
        cv[y0_:y1_]+=ov*env+cv2.GaussianBlur(ov,(0,0),5)*0.5*env

# ---------------- sprites / particles
def spr(r):
    s=int(r*6)|1; c=s//2
    y,x=np.mgrid[0:s,0:s]; g=np.exp(-((x-c)**2+(y-c)**2)/(2*(r**2)))
    return g.astype(np.float32)
SPR={r:spr(r) for r in (1.6,2.6,4,22,40)}
def add_sprite(canvas,spr_,x,y,col,a):
    s=spr_.shape[0]; h=s//2; x=int(x); y=int(y)
    xa,ya=max(0,x-h),max(0,y-h); xb,yb=min(W,x-h+s),min(H,y-h+s)
    if xa>=xb or ya>=yb: return
    sp=spr_[ya-(y-h):yb-(y-h), xa-(x-h):xb-(x-h)]
    canvas[ya:yb,xa:xb]+=sp[...,None]*(col*a)
NP=110
pk=[]
for i in range(NP):
    big=i<8
    pk.append(dict(x=rng.uniform(0,W),y=rng.uniform(0,H),v=rng.uniform(8,30)*(0.5 if big else 1),
        r=(40 if i<3 else 22) if big else rng.choice([1.6,2.6,4]),ph=rng.uniform(0,6.28),
        a=(0.12 if big else rng.uniform(0.35,1.0)),dx=rng.uniform(-8,8),big=big))
grain=[cv2.resize((rng.randn(H//2,W//2).astype(np.float32)*5),(W,H))[...,None] for _ in range(6)]
SPK=[(rng.uniform(0,2*math.pi),rng.uniform(90,420),rng.uniform(0.7,1.5),rng.choice([1.6,2.6,4])) for _ in range(70)]
def draw_sparks(cv,t,t0,px,py):
    k=t-t0
    if k<0 or k>1.6: return
    for a_,v,life,r in SPK:
        if k>life: continue
        x=px+math.cos(a_)*v*k*(1-0.35*k); y=py+math.sin(a_)*v*k*(1-0.35*k)+120*k*k
        add_sprite(cv,SPR[r],x,y,GOLD,(1-k/life)**1.5*1.4)
def draw_shock(cv,t,t0,cx,cy):
    k=t-t0
    if k<0 or k>1.0: return
    r=int(60+900*eo3(k)); al=(1-k)**2
    ov=np.zeros((H//2,W//2,3),np.float32)
    cv2.circle(ov,(cx//2,cy//2),r//2,(255,215,140),max(2,int(26*(1-k))),cv2.LINE_AA)
    cv+=cv2.resize(cv2.GaussianBlur(ov,(0,0),6),(W,H))*(0.55*al)

# ---------------- coin renderer
def coin_patch(theta,D,shine):
    c=math.cos(theta); sn=math.sin(theta)
    D=int(round(D)); w=max(int(D*abs(c)),3)
    face=cv2.resize(coinA,(max(w,2),D),interpolation=cv2.INTER_AREA if D<SD else cv2.INTER_LINEAR)
    rgb=face[...,:3].copy(); a=face[...,3]
    if shine is not None:
        band=np.exp(-((shine_u-shine)/0.07)**2)*coin_mask
        rgb=rgb+cv2.resize(band.astype(np.float32),(max(w,2),D))[...,None]*np.array([255,235,190],np.float32)*0.55
    rgb*=(0.62+0.38*abs(c))
    edge=int(round(D*0.045*abs(sn)))
    if edge>0:
        tw=w+edge
        patch=np.zeros((D,tw,4),np.float32)
        ed=np.zeros((D,tw),np.uint8)
        cv2.ellipse(ed,(tw//2,D//2),(tw//2,D//2),0,0,360,255,-1,cv2.LINE_AA)
        gradx=np.linspace(0,1,tw)[None,:]*np.ones((D,1))
        patch[...,:3]=np.stack([130+60*gradx,95+45*gradx,22+10*gradx],-1); patch[...,3]=ed/255.0
        off=-edge//2 if c*sn>0 else edge//2
        fx=min(max((tw-w)//2+off,0),tw-w)
        reg=patch[:,fx:fx+w]; fa=a[:,:w][...,None]
        reg[...,:3]=rgb[:,:w]*fa+reg[...,:3]*(1-fa); reg[...,3]=np.maximum(reg[...,3],a[:,:w])
        return patch
    return np.dstack([rgb[:,:w],a[:,:w]])
def blit(canvas,patch,cx,cy,alpha=1.0):
    h,w=patch.shape[:2]; x=int(round(cx-w/2)); y=int(round(cy-h/2))
    xa,ya=max(0,x),max(0,y); xb,yb=min(W,x+w),min(H,y+h)
    if xa>=xb or ya>=yb: return
    p=patch[ya-y:yb-y, xa-x:xb-x]; al=p[...,3:4]*alpha
    canvas[ya:yb,xa:xb]=canvas[ya:yb,xa:xb]*(1-al)+p[...,:3]*al

# ---------------- scene A (intro stage)
cxs,cys=540,820
FLOOR=1230
bgA=np.zeros((H,W,3),np.float32)
d=np.hypot((xx-cxs)*1.2,(yy-cys))/1100
glow=np.clip(1-d,0,1)**1.8
bgA[...,0]=4+16*glow; bgA[...,1]=9+44*glow; bgA[...,2]=20+78*glow
fl=np.clip((yy-FLOOR)/(H-FLOOR),0,1)
bgA*=(1-0.35*fl[...,None])
fg=np.exp(-((xx-cxs)/480)**2)*np.clip(1-abs(yy-FLOOR-60)/260,0,1)*(yy>FLOOR)
bgA[...,0]+=14*fg; bgA[...,1]+=11*fg
cv2.line(bgA,(0,FLOOR),(W,FLOOR),(60,75,95),1)
bgA=cv2.GaussianBlur(bgA,(0,0),1.0)
hw,hh=W//2,H//2
hy,hx=np.mgrid[0:hh,0:hw].astype(np.float32)
ang=np.arctan2(hy-cys/2,hx-cxs/2); rad=np.hypot(hx-cxs/2,hy-cys/2)
rayfall=np.exp(-rad/380).astype(np.float32)
sxs=np.arange(W)-W/2
streak=(np.exp(-(np.arange(-40,41)/3.2)**2)[:,None]*np.exp(-np.abs(sxs)/250)[None,:]).astype(np.float32)

# ---------------- scene B (course card)
bgB=np.zeros((H,W,3),np.float32)
dB=np.hypot((xx-540)*0.9,(yy-900)*0.62)/900
gB=np.clip(1-dB,0,1)**1.5
bgB[...,0]=6+12*gB; bgB[...,1]=15+42*gB; bgB[...,2]=34+66*gB
# grid pattern
gridl=np.zeros((H,W),np.float32)
for gx_ in range(0,W,135): gridl[:,gx_:gx_+1]=1
for gy_ in range(0,H,135): gridl[gy_:gy_+1,:]=1
bgB+=gridl[...,None]*np.array([6,14,22],np.float32)*(0.4+0.6*gB[...,None])
# skyline
BASE=1585; RY0=1290; RH=BASE+30-RY0
bl=[]
x=-10
heights=[130,200,150,95,250,170,120,210,145,100,175,135]
i=0
while x<W+20:
    w=int(rng.uniform(78,128)); h=heights[i%len(heights)]+int(rng.uniform(-15,25)); bl.append((x,w,h)); x+=w+int(rng.uniform(4,14)); i+=1
sky_body=np.zeros((RH,W,4),np.float32)
win_dim=np.zeros((RH,W,4),np.float32); win_bright=np.zeros((RH,W,4),np.float32)
for (bx,bw,bh) in bl:
    top=BASE-bh-RY0
    cv2.rectangle(sky_body,(bx,top),(bx+bw,BASE-RY0+30),(10,26,46,1.0),-1)
    cv2.rectangle(sky_body,(bx,top),(bx+bw,top+3),(40,80,120,1.0),-1)
    for wy in range(top+16,BASE-RY0-8,30):
        for wx in range(bx+12,bx+bw-14,22):
            r_=rng.rand()
            if r_<0.55:
                cv2.rectangle(win_dim,(wx,wy),(wx+9,wy+14),(205,170,95,0.35),-1)
                cv2.rectangle(win_bright,(wx,wy),(wx+9,wy+14),(255,215,120,1.0),-1)
            elif r_<0.8:
                cv2.rectangle(win_bright,(wx,wy),(wx+9,wy+14),(235,195,105,0.85),-1)
# brace segments for tall buildings
segs=[]
tall=sorted(bl,key=lambda b:-b[2])[:4]
for (bx,bw,bh) in tall:
    floors=int((bh-20)//58)
    for f in range(min(floors,3)):
        yb_=BASE-RY0-10-f*58; yt_=yb_-52
        segs.append(((bx+6,yb_),(bx+bw-6,yt_))); segs.append(((bx+bw-6,yb_),(bx+6,yt_)))
NSEG=len(segs)
def sky_layer(t,shear_amp,phase,bracep,winb):
    lay=sky_body.copy()
    wd=win_dim*(1-winb)+win_bright*winb
    a=wd[...,3:4]; lay[...,:3]=lay[...,:3]*(1-a)+wd[...,:3]*a
    lay[...,3]=np.maximum(lay[...,3],wd[...,3])
    # braces
    if bracep>0:
        n=bracep*NSEG
        for i,(p0,p1) in enumerate(segs):
            f=np.clip(n-i,0,1)
            if f<=0: break
            q=(int(p0[0]+(p1[0]-p0[0])*f),int(p0[1]+(p1[1]-p0[1])*f))
            cv2.line(lay,p0,q,(255,205,95,1.0),4,cv2.LINE_AA)
    # shear
    if shear_amp>0.05:
        ys=np.arange(RH,dtype=np.float32)
        hfac=np.clip((BASE-RY0-ys)/260.0,0,1.2)
        dx=shear_amp*phase*hfac
        mapx=(np.arange(W,dtype=np.float32)[None,:]+dx[:,None]).astype(np.float32)
        mapy=np.repeat(ys[:,None],W,1).astype(np.float32)
        lay=cv2.remap(lay,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    return lay
def quake_env(t): return sstep((t-6.6)/0.4)*(1-sstep((t-8.7)/1.5))
def draw_frame(cv,p):
    if p<=0: return
    ov=np.zeros((H//2,W//2,3),np.float32)
    col=(235,190,95)
    for ins,th,pp in ((26,3,p),(44,1,p)):
        pts=[(W/2,ins),(W-ins,ins),(W-ins,H-ins),(ins,H-ins),(ins,ins),(W/2,ins)]
        segl=[math.hypot(pts[i+1][0]-pts[i][0],pts[i+1][1]-pts[i][1]) for i in range(5)]
        tot=sum(segl)
        # clockwise from top-center to p*tot/2, counter to same
        for dirn in (1,-1):
            order=range(5) if dirn==1 else range(4,-1,-1)
            remain=pp*tot/2*1.0
            # path param: first half runs top-center -> ... meet bottom center
            L=0
            ptsd=pts if dirn==1 else [pts[0]]+[pts[5-i] for i in range(1,5)]+[pts[0]]
            # build counter-clockwise explicit
            if dirn==-1: ptsd=[(W/2,ins),(ins,ins),(ins,H-ins),(W-ins,H-ins),(W-ins,ins),(W/2,ins)]
            for i in range(5):
                a=ptsd[i]; b=ptsd[i+1]; sl=math.hypot(b[0]-a[0],b[1]-a[1])
                if remain<=0: break
                f=min(1.0,remain/sl); q=(a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f)
                cv2.line(cv,(int(a[0]),int(a[1])),(int(q[0]),int(q[1])),col,th,cv2.LINE_AA)
                cv2.line(ov,(int(a[0]/2),int(a[1]/2)),(int(q[0]/2),int(q[1]/2)),col,th,cv2.LINE_AA)
                remain-=sl
    if p>0.97:
        for (px_,py_) in ((26,26),(W-26,26),(W-26,H-26),(26,H-26)):
            pts=np.array([[px_,py_-16],[px_+16,py_],[px_,py_+16],[px_-16,py_]],np.int32)
            cv2.fillPoly(cv,[pts],(245,205,120),cv2.LINE_AA)
            cv2.fillPoly(ov,[pts//2],(245,205,120),cv2.LINE_AA)
    cv+=cv2.resize(cv2.GaussianBlur(ov,(0,0),4),(W,H))*0.6
def wave(t,amp):
    xs=np.arange(70,1011,5,dtype=np.float32)
    y=(np.sin(xs*0.045+t*17)*0.6+np.sin(xs*0.11-t*23+1.3)*0.3+np.sin(xs*0.021+t*9.0)*0.5+np.sin(xs*0.33+t*31)*0.12)
    env=np.exp(-((xs-540)/520)**2*0.7)
    return xs,(y*env*amp).astype(np.float32)

T_APP=0.8; T_SPIN=3.4; T_SETTLE=4.2
T_FLY0=5.0; T_FLY1=6.4
FINAL=(540,420,300)   # coin final (cx,cy,D)
def renderB(t,pb):
    cv=bgB.copy()
    # ambient pulsing glow behind titles
    if t>6.5:
        pulse=0.5+0.5*math.sin((t-6.5)*1.6)
        gg=np.exp(-(np.hypot((xx-540)*0.9,(yy-800)*0.7)/420)**1.5)
        cv+=gg[...,None]*np.array([22,34,52],np.float32)*(0.5+0.5*sstep((t-6.5)/1.5))*(0.8+0.2*pulse)
    draw_frame(cv,eo3((t-5.3)/1.7))
    # skyline + quake
    q=quake_env(t)
    shear=26*q
    phase=math.sin(2*math.pi*3.1*t)+0.5*math.sin(2*math.pi*5.3*t+1.0)
    bracep=eo3((t-8.0)/1.8)
    winb=sstep((t-9.6)/0.9)
    lay=sky_layer(t,shear,phase,bracep,winb)
    skyop=0.95*sstep((t-5.6)/0.8)
    reg=cv[RY0:RY0+RH]
    a=lay[...,3:4]*skyop
    reg[:]=reg*(1-a)+lay[...,:3]*a
    # stable gold edge glow after retrofit
    if t>9.8:
        gl=sstep((t-9.8)/0.8)*(0.7+0.3*math.sin((t-9.8)*2))
        e=cv2.GaussianBlur(lay[...,3]*sstep(1)*0+lay[...,:3].sum(-1)*0,(0,0),1)
    # seismograph
    ybase=1470
    amp=70*q+ (3.0+2.0*math.sin(t*2.0))*sstep((t-6.4)/0.6)
    xs,ys=wave(t,amp if t>6.4 else 0)
    if t>5.8:
        env=sstep((t-5.8)/0.6)
        y0_,y1_=ybase-120,ybase+120
        ov=np.zeros((y1_-y0_,W,3),np.float32)
        pts=np.stack([xs,ybase-y0_-ys],1).astype(np.int32)
        col=(120,220,255) if q>0.15 else (255,205,110)
        cv2.polylines(ov,[pts],False,col,3,cv2.LINE_AA)
        cv[y0_:y1_]+=ov*env+cv2.GaussianBlur(ov,(0,0),7)*0.9*env
        cv2.line(cv,(70,ybase),(1010,ybase),(40,70,100),1,cv2.LINE_AA)
    return cv
def zoomB(t): return 1.0
def render(t):
    cv=bgA.copy()
    renv=sstep((t-0.2)/1.0)*(1-sstep((t-4.6)/1.2))
    if renv>0.01:
        rot=0.35*t
        rr=(np.abs(np.sin(7*ang+rot))**6*0.75+np.abs(np.sin(11*ang-rot*1.3))**9*0.55+np.abs(np.sin(4*ang+rot*0.6))**4*0.25)*rayfall
        cv+=cv2.resize(rr,(W,H))[...,None]*GOLD*(0.30*renv)
    u=(t-T_APP)/T_SPIN
    theta=2*math.pi*3*eo3(u) if t>=T_APP else 0
    sc=0.45+0.55*eo3((t-T_APP)/1.1)
    D0=700
    appear=sstep((t-T_APP)/0.45)
    bob=6*math.sin(2*math.pi*(t-T_APP)*0.45)*sstep((t-T_APP)/1.0)*(1-sstep((t-T_FLY0)/0.4)) if t>=T_APP else 0
    ccx,ccy=cxs,cys+bob+(1-eo3((t-T_APP)/1.1))*40
    genv=appear*(1-sstep((t-5.0)/1.0))
    if genv>0.01:
        gl=np.exp(-(np.hypot(xx-ccx,yy-ccy)/400.0)**1.6).astype(np.float32)
        cv+=gl[...,None]*GOLD*(0.28*genv)
    shine=None
    if T_SETTLE<=t<=T_SETTLE+0.9: shine=-0.3+1.6*eio3((t-T_SETTLE)/0.9)
    patch=None
    if appear>0.001 and t<T_FLY0:
        Dc=D0*sc
        patch=coin_patch(theta,Dc,shine)
        ph,pw=patch.shape[:2]
        yb=ccy+Dc/2; top_r=2*FLOOR-yb
        refl=patch[::-1].copy()
        refl[...,3]*=np.linspace(0.38,0.0,ph)[:,None]*appear*(1-sstep((t-T_FLY0+0.2)/0.6))
        refl=cv2.GaussianBlur(refl,(0,0),2.2); refl[...,:3]*=0.8
        blit(cv,refl,ccx,top_r+ph/2)
    pb=sstep((t-(T_FLY0+0.1))/1.0)
    if pb>0:
        B=renderB(t,pb)
        cv=cv*(1-pb)+B*pb
    # coin
    if t>=T_FLY0:
        u2=eio3((t-T_FLY0)/(T_FLY1-T_FLY0))
        tx,ty,tD=FINAL
        px=cxs+(tx-cxs)*u2; py=cys+(ty-cys)*u2-90*math.sin(math.pi*u2)
        Dc=D0+(tD-D0)*u2
        # halo behind coin
        halo=0.28+0.12*math.sin(t*2.0)
        if t>T_FLY1: add_sprite(cv,cv2.resize(SPR[40],(560,560)),px,py,np.array([255,190,100],np.float32),halo*sstep((t-T_FLY1)/0.5))
        blit(cv,coin_patch(0.0,Dc,None),px,py,1.0)
    elif patch is not None:
        blit(cv,patch,ccx,ccy,appear)
    # flares & effects
    f1=math.exp(-((t-0.9)/0.22)**2)*0.9
    f2=math.exp(-((t-T_SETTLE)/0.28)**2)
    f3=math.exp(-((t-6.45)/0.25)**2)*0.8
    for fv,(fx,fy) in ((f1,(cxs,cys)),(f2,(cxs,cys)),(f3,(FINAL[0],FINAL[1]))):
        if fv>0.02:
            y0_=int(fy)-40
            M=np.float32([[1,0,fx-W/2],[0,1,0]])
            s=cv2.warpAffine(streak,M,(W,81),flags=cv2.INTER_LINEAR)
            ya=max(0,y0_); yb=min(H,y0_+81)
            cv[ya:yb]+=s[ya-y0_:yb-y0_,:,None]*np.array([255,215,150],np.float32)*(0.9*fv)
            add_sprite(cv,cv2.resize(SPR[40],(460,460)),fx,fy,np.array([255,200,110],np.float32),0.45*fv)
    draw_shock(cv,t,T_SETTLE,cxs,cys)
    draw_sparks(cv,t,6.4,FINAL[0],FINAL[1])
    # ---- texts: phase 1 (organizer)
    draw_text(cv,'a1',t,3.2,5.0,540,1440)
    draw_text(cv,'a2',t,3.5,5.0,540,1515)
    draw_line(cv,t,3.9,5.0,1385,300)
    # ---- phase 2 titles
    draw_text(cv,'label',t,6.0,99,540,655)
    draw_text(cv,'t1',t,6.5,99,540,795,dur=1.2)
    draw_text(cv,'t2',t,6.9,99,540,930,dur=1.2)
    draw_line(cv,t,7.6,99,1012,330)
    draw_text(cv,'with',t,7.9,99,540,1078)
    draw_text(cv,'dr',t,8.3,99,540,1168,dur=1.3)
    draw_line(cv,t,9.2,99,1250,330)
    draw_text(cv,'org1',t,9.5,99,540,1318)
    draw_text(cv,'org2',t,9.8,99,540,1378)
    # particles
    for p in pk:
        y_=(p['y']-p['v']*t)%H; x_=(p['x']+p['dx']*t+14*math.sin(t*0.6+p['ph']))%W
        tw=0.6+0.4*math.sin(t*3+p['ph']*3); a=p['a']*tw
        if p['big']: a=p['a']*(0.7+0.3*math.sin(t*0.7+p['ph']))
        add_sprite(cv,SPR[p['r']],x_,y_,GOLD,a*sstep(t/1.2))
    # glint sweep over everything bright
    gs=sstep((t-10.6)/0.1)*(1-sstep((t-12.2)/0.1))
    if gs>0:
        pos=-300+1900*eio3((t-10.6)/1.6)
        band=np.exp(-((xx*0.6+yy*0.45-pos*0.9-300)/90.0)**2).astype(np.float32)
        lum=np.clip(cv.mean(-1)/255.0,0,1)**2.2
        cv+=(band*lum*1.2)[...,None]*np.array([255,230,170],np.float32)*gs
    # shakes
    sh=np.zeros(2)
    if T_SETTLE<=t<T_SETTLE+0.7:
        k=t-T_SETTLE; amp=11*math.exp(-k*7); sh+=(amp*math.sin(k*70),amp*math.cos(k*61))
    if 6.6<=t<8.8:
        k=t-6.6; amp=5*quake_env(t); sh+=(amp*math.sin(k*47),amp*math.cos(k*53))
    if abs(sh).sum()>0.01:
        cv=cv2.warpAffine(cv,np.float32([[1,0,sh[0]],[0,1,sh[1]]]),(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    vig=np.clip(1-(np.hypot((xx-W/2)/(W*0.75),(yy-H/2)/(H*0.62)))**2.2*0.7,0.25,1)[...,None]
    cv*=vig
    cv+=grain[int(t*FPS)%6]*0.6
    cv*=1-sstep((t-(TEND-0.5))/0.5)
    return np.clip(cv,0,255).astype(np.uint8)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='test':
        for tt in [float(a) for a in sys.argv[2:]]:
            Image.fromarray(render(tt)).save(f'/tmp/v3_{tt:.2f}.png')
    else:
        n=int(TEND*FPS)
        ff=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-preset','medium','-crf','17','-pix_fmt','yuv420p','/home/claude/video3_noaudio.mp4'],stdin=subprocess.PIPE,stderr=open('/tmp/ff3.log','w'))
        for i in range(n):
            ff.stdin.write(render(i/FPS).tobytes())
            if i%15==0: open('/tmp/prog3.txt','w').write(f'{i}/{n}')
        ff.stdin.close(); ff.wait(); open('/tmp/prog3.txt','w').write('done')
