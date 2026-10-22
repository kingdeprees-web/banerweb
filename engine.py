import cv2, numpy as np, math, sys, subprocess
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os, json, tempfile
HERE=os.path.dirname(os.path.abspath(__file__))
CFG=json.load(open(os.environ['INTRO_CFG'],encoding='utf-8'))
import intro3 as I
I.FB=os.path.join(HERE,'assets','B-NAZANIN.TTF')
def _render_text(txt,size,top,bot):
    font=ImageFont.truetype(I.FB,size,layout_engine=ImageFont.Layout.RAQM)
    sw=max(1,int(size*0.028))
    wlen=int(font.getlength(txt,direction='rtl',language='fa'))+size
    hh=int(size*2.2)
    im=Image.new('L',(wlen,hh),0)
    ImageDraw.Draw(im).text((wlen//2,hh//2),txt,font=font,fill=255,anchor='mm',direction='rtl',language='fa',stroke_width=sw,stroke_fill=255)
    a=np.array(im).astype(np.float32)/255
    ys,xs=np.where(a>0.02); pad=8
    a=a[max(0,ys.min()-pad):ys.max()+pad, max(0,xs.min()-pad):xs.max()+pad]
    h,w=a.shape; g=np.linspace(0,1,h)[:,None,None]
    rgb=(np.array(top,np.float32)*(1-g)+np.array(bot,np.float32)*g)*np.ones((1,w,1),np.float32)
    return rgb,a
I.render_text=_render_text
from intro3 import sstep,eo3,eio3,eisine,W,H,FPS,GOLD
xx,yy=I.xx,I.yy
TEND=0
def eob(x):
    x=min(max(x,0),1); c1=1.70158; c3=c1+1
    return 1+c3*(x-1)**3+c1*(x-1)**2

# ---------------- texts
def fit_prep(name,txt,start,maxw,top,bot,col):
    size=start
    while size>20:
        f=ImageFont.truetype(I.FB,size,layout_engine=ImageFont.Layout.RAQM)
        if f.getlength(txt,direction='rtl',language='fa')<=maxw: break
        size-=4
    I.prep(name,txt,size,top,bot,col)
GT,GB=I.GT,I.GB; WT,WB=I.WT,I.WB
CY=((200,230,255),(170,205,240))
fit_prep('a1','انجمن صنفی کارفرمایی سازندگان',54,700,WT,WB,(255,215,150))
fit_prep('a2','ذی‌صلاح ساختمان شهرستان ارومیه',54,700,WT,WB,(255,215,150))
fit_prep('label','اعضای کمیته تخصصی',84,900,WT,WB,(255,215,150))
fit_prep('title',CFG['title'],150,900,GT,GB,(255,200,100))
MEM=[(m['n'],m['c'],m.get('r') or None) for m in CFG['members']]
for i,(n,c,r) in enumerate(MEM):
    fit_prep(f'n{i}',n,112,900,GT,GB,(255,200,100))
    fit_prep(f'c{i}',c,70,900,WT,WB,(255,215,150))
    if r: fit_prep(f'r{i}','('+r+')',60,900,(255,232,160),(224,170,70),(255,200,100))

# ---------------- banner + avatars
BANFILE=CFG['banner']
BAN=np.array(Image.open(BANFILE).convert('RGB')).astype(np.float32)
BW0,BH0=BAN.shape[1],BAN.shape[0]
TALL=BH0/BW0>1.9
SV=960.0/BW0 if TALL else min(960.0/BW0,1330.0/BH0)
VW=int(round(BW0*SV)); VH=1330 if TALL else int(round(BH0*SV))
VX0=(W-VW)//2; VY0=250 if TALL else int(915-VH/2)
BANS=cv2.resize(BAN,(VW,int(round(BH0*SV))),interpolation=cv2.INTER_AREA)
SCROLL_MAX=max(0,BANS.shape[0]-VH)
CIR=CFG['circles']; N=len(CIR); LOGO=CFG['logo']
T_M0=6.9; DT=1.0
T_P4=T_M0+DT*N+0.2; TB=T_P4+0.6
if TALL: S0=TB+2.9; S1=S0+DT*N; TEND=S1+2.9
else: S0=S1=TB+3.0; TEND=TB+9.8
def scroll(t): return SCROLL_MAX*eisine((t-S0)/(S1-S0)) if SCROLL_MAX>0 else 0
def slot(i,s=None,cx=None,cy=None): return VX0+CIR[i]['x']*SV, VY0+CIR[i]['y']*SV
def logo_slot(s=None,cx=None,cy=None): return VX0+LOGO['x']*SV, VY0+LOGO['y']*SV
VISIBLE=[(CIR[i]['y']*SV>40 and CIR[i]['y']*SV<VH-40) for i in range(N)]
NVIS=sum(VISIBLE)
DSLOT=2*0.922*float(np.mean([c['r'] for c in CIR]))*SV
def _rr(w,h,r,ss=3):
    im=Image.new('L',(w*ss,h*ss),0); ImageDraw.Draw(im).rounded_rectangle((0,0,w*ss-1,h*ss-1),radius=r*ss,fill=255)
    return np.array(im.resize((w,h),Image.LANCZOS)).astype(np.float32)/255
VMASK=_rr(VW,VH,30)
_fy=np.minimum(np.arange(VH),VH-1-np.arange(VH)).astype(np.float32)
if TALL: VMASK=VMASK*np.clip(_fy/30.0,0,1)[:,None]
_bo=_rr(VW+12,VH+12,34); _bi=np.zeros_like(_bo); _bi[5:5+_rr(VW+2,VH+2,30).shape[0],5:5+_rr(VW+2,VH+2,30).shape[1]]=_rr(VW+2,VH+2,30)
_bd=np.clip(_bo-_bi,0,1)
BORDER=np.zeros((VH+12,VW+12,4),np.float32)
_g=np.linspace(0,1,VH+12)[:,None,None]
BORDER[...,:3]=(np.array([255,228,150],np.float32)*(1-_g)+np.array([196,140,42],np.float32)*_g)*np.ones((1,VW+12,1),np.float32)
BORDER[...,3]=_bd
srcim=Image.open(BANFILE).convert('RGB')
SP=min(190,int(900/(N-1))) if N>1 else 0
ROWD=int(min(112,SP*0.77)) if N>1 else 112
AV=[]
for i in range(N):
    cx_=CIR[i]['x']; cy_=CIR[i]['y']; r=int(round(CIR[i]['r']*0.99))
    crop=srcim.crop((int(round(cx_-r)),int(round(cy_-r)),int(round(cx_+r)),int(round(cy_+r)))).resize((440,440),Image.LANCZOS)
    crop=crop.filter(ImageFilter.UnsharpMask(radius=2.2,percent=110,threshold=2))
    m=Image.new('L',(440*2,440*2),0); RM=71*440/154.0
    ImageDraw.Draw(m).ellipse((2*(220-RM),2*(220-RM),2*(220+RM),2*(220+RM)),fill=255)
    a=np.array(m.resize((440,440),Image.LANCZOS)).astype(np.float32)/255
    rgb=np.array(crop).astype(np.float32)
    AV.append(np.dstack([rgb,a]))
def avatar_patch(i,Din,ring_p,glow=0.0):
    Din=int(round(Din)); S=Din+40
    p=np.zeros((S,S,4),np.float32)
    av=cv2.resize(AV[i],(int(Din*77/71),)*2,interpolation=cv2.INTER_AREA if Din<440 else cv2.INTER_LINEAR)
    h=av.shape[0]; o=(S-h)//2
    p[o:o+h,o:o+h]=av
    ring=np.zeros((S,S),np.uint8)
    rr=int(Din/2+9)
    if ring_p>0:
        cv2.ellipse(ring,(S//2,S//2),(rr,rr),0,-90,-90+360*min(ring_p,1)-0.01,255,5,cv2.LINE_AA)
    rg=ring.astype(np.float32)/255
    g=np.linspace(0,1,S)[:,None]
    col=np.stack([255-55*g,226-86*g,140-100*g],-1)*np.ones((1,S,1),np.float32)
    p[...,:3]=p[...,:3]*(1-rg[...,None])+col*rg[...,None]
    p[...,3]=np.maximum(p[...,3],rg)
    return p
def slot_ring(S,Din,alpha):
    # faint placeholder ring
    S=int(Din+40); r=np.zeros((S,S),np.uint8)
    cv2.ellipse(r,(S//2,S//2),(int(Din/2+9),int(Din/2+9)),0,0,360,255,3,cv2.LINE_AA)
    p=np.zeros((S,S,4),np.float32); p[...,:3]=(235,190,95); p[...,3]=r/255.0*alpha
    return p

# ---------------- backgrounds
cxs,cys=I.cxs,I.cys; FLOOR=I.FLOOR
bgA=np.zeros((H,W,3),np.float32)
d=np.hypot((xx-cxs)*1.2,(yy-cys))/1100
glow=np.clip(1-d,0,1)**1.8
bgA[...,0]=3+34*glow; bgA[...,1]=3+22*glow; bgA[...,2]=4+7*glow
fl=np.clip((yy-FLOOR)/(H-FLOOR),0,1)
bgA*=(1-0.35*fl[...,None])
fg=np.exp(-((xx-cxs)/480)**2)*np.clip(1-abs(yy-FLOOR-60)/260,0,1)*(yy>FLOOR)
bgA[...,0]+=16*fg; bgA[...,1]+=11*fg
cv2.line(bgA,(0,FLOOR),(W,FLOOR),(80,64,40),1)
bgA=cv2.GaussianBlur(bgA,(0,0),1.0)
# ---------- new opening assets
bgN=np.zeros((H,W,3),np.float32)
dN=np.hypot((xx-cxs)*1.1,(yy-cys))/1150; gN=np.clip(1-dN,0,1)**1.7
bgN[...,0]=3+30*gN; bgN[...,1]=3+20*gN; bgN[...,2]=4+7*gN
_rng=np.random.RandomState(21)
_pts=np.argwhere(I.coinA[...,3]>0.5)
_sel=_rng.choice(len(_pts),22000,replace=False); _pts=_pts[_sel]
_s=700/(2*I.R0)
_tx=(_pts[:,1]-I.SD/2)*_s; _ty=(_pts[:,0]-I.SD/2)*_s
PTH=np.arctan2(_ty,_tx).astype(np.float32); PRT=np.hypot(_tx,_ty).astype(np.float32)
PR0=_rng.uniform(420,1500,len(_pts)).astype(np.float32)
PDELTA=_rng.uniform(2.2,6.0,len(_pts)).astype(np.float32)
PD_=_rng.uniform(0,0.4,len(_pts)).astype(np.float32)
_col=I.coinA[_pts[:,0],_pts[:,1],:3]
PCOL=[np.clip(_col[:,c]*1.1+25,0,255)*0.85 for c in range(3)]
bgB=np.zeros((H,W,3),np.float32)
dB=np.hypot((xx-540)*0.9,(yy-860)*0.62)/950
gB=np.clip(1-dB,0,1)**1.6
bgB[...,0]=4+22*gB; bgB[...,1]=4+15*gB; bgB[...,2]=5+5*gB
gl_=np.zeros((H,W),np.float32)
for gx_ in range(0,W,135): gl_[:,gx_:gx_+1]=1
for gy_ in range(0,H,135): gl_[gy_:gy_+1,:]=1
bgB+=gl_[...,None]*np.array([5,4,1.5],np.float32)*(0.3+0.7*gB[...,None])

T_FLY0,T_FLY1=5.0,6.4
FIN=(540,350,250)
T_M0=6.9; DT=1.0
ROW_Y=1470
def row_slot(i): return 540+((N-1)/2-i)*SP, ROW_Y
def coin_state(t):
    if t<T_FLY0: return None
    if t<T_P4:
        u=eio3((t-T_FLY0)/(T_FLY1-T_FLY0))
        return (cxs+(FIN[0]-cxs)*u, cys+(FIN[1]-cys)*u-90*math.sin(math.pi*u), 700+(FIN[2]-700)*u, 1.0)
    u=eio3((t-T_P4)/1.3); lx,ly=logo_slot()
    al=1-sstep((t-(TB+1.6))/0.5)
    return (FIN[0]+(lx-FIN[0])*u, FIN[1]+(ly-FIN[1])*u-70*math.sin(math.pi*u), FIN[2]+(2*LOGO['r']*SV-FIN[2])*u, al)

def banner_scale(t): return SB if t<13.5 else SB*(1+0.02*eisine((t-13.5)/7.0))

def add_glow(cv,cx,cy,w,h,col,a):
    g=cv2.resize(I.SPR[40],(int(w),int(h)))
    x=int(cx-w/2); y=int(cy-h/2)
    xa,ya=max(0,x),max(0,y); xb,yb=min(W,x+int(w)),min(H,y+int(h))
    if xa>=xb or ya>=yb: return
    cv[ya:yb,xa:xb]+=g[ya-y:yb-y,xa-x:xb-x,None]*(col*a)

def render(t):
    cv=bgA.copy()
    renv=sstep((t-0.2)/1.0)*(1-sstep((t-4.6)/1.2))
    if renv>0.01:
        rot=0.35*t
        rr=(np.abs(np.sin(7*I.ang+rot))**6*0.75+np.abs(np.sin(11*I.ang-rot*1.3))**9*0.55+np.abs(np.sin(4*I.ang+rot*0.6))**4*0.25)*I.rayfall
        cv+=cv2.resize(rr,(W,H))[...,None]*GOLD*(0.30*renv)
    T_APP,T_SPIN,T_SETTLE=0.8,3.4,4.2
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
        patch=I.coin_patch(theta,Dc,shine)
        ph=patch.shape[0]; yb=ccy+Dc/2; top_r=2*FLOOR-yb
        refl=patch[::-1].copy()
        refl[...,3]*=np.linspace(0.38,0.0,ph)[:,None]*appear*(1-sstep((t-T_FLY0+0.2)/0.6))
        refl=cv2.GaussianBlur(refl,(0,0),2.2); refl[...,:3]*=0.8
        I.blit(cv,refl,ccx,top_r+ph/2)
    # ---- scene B
    pb=sstep((t-(T_FLY0+0.1))/1.0)
    if pb>0:
        B=bgB.copy()
        if t>6.0:
            pulse=0.5+0.5*math.sin((t-6.0)*1.5)
            gg=np.exp(-(np.hypot((xx-540)*0.9,(yy-860)*0.7)/430)**1.5)
            B+=gg[...,None]*np.array([30,22,8],np.float32)*(0.5+0.5*sstep((t-6.0)/1.5))*(0.8+0.2*pulse)
        I.draw_frame(B,eo3((t-5.3)/1.7))
        # banner viewport (tall banner scrolls)
        if t>=TB-0.1:
            sc_=int(scroll(t))
            band_=BANS[sc_:sc_+VH]
            p=eo3((t-TB)/1.4)
            ys_=np.arange(VH,dtype=np.float32)
            front=(VH+160)*p
            mrow=sstep((front-ys_)/160)*sstep((t-(TB-0.1))/0.3)
            a_=(VMASK*mrow[:,None])[...,None]
            B[VY0:VY0+VH,VX0:VX0+VW]=B[VY0:VY0+VH,VX0:VX0+VW]*(1-a_)+band_*a_
            if p<0.999:
                ly=int(VY0+front-80)
                if VY0<=ly<VY0+VH:
                    bnd=np.exp(-((yy-ly)/16.0)**2)[...,None]*np.array([255,215,140],np.float32)*0.6
                    ins=((xx>=VX0)&(xx<VX0+VW)).astype(np.float32)[...,None]
                    B+=bnd*ins
            bo_=BORDER; bal=sstep((t-TB)/0.5)
            B[VY0-6:VY0+VH+6,VX0-6:VX0+VW+6]=B[VY0-6:VY0+VH+6,VX0-6:VX0+VW+6]*(1-bo_[...,3:4]*bal)+bo_[...,:3]*bo_[...,3:4]*bal
        cv=cv*(1-pb)+B*pb
    # ---- slot placeholders + avatars
    if t>6.6:
        for i in range(N):
            rx,ry=row_slot(i)
            ph=sstep((t-6.6-0.08*i)/0.5)*(1-sstep((t-T_P4)/0.3))
            if ph>0.01: I.blit(cv,slot_ring(0,ROWD,0.35),rx,ry,ph)
    for i in range(N):
        ti=T_M0+DT*i; a=t-ti
        if a<0: continue
        rx,ry=row_slot(i)
        if a<0.8:
            D=270*(0.55+0.45*eob(min(a/0.45,1))); cx_,cy_=540,980; al=sstep(a/0.25); rp=eo3(a/0.55)
            if a<0.8 and a>0.05:
                add=I.add_sprite
                add(cv,cv2.resize(I.SPR[40],(int(D*1.9),)*2),cx_,cy_,np.array([255,190,100],np.float32),0.30*al)
        elif a<1.12:
            u2=eio3((a-0.8)/0.32); cx_=540+(rx-540)*u2; cy_=980+(ry-980)*u2; D=270+(ROWD-270)*u2; al=1; rp=1
        else:
            cx_,cy_,D,al,rp=rx,ry,ROWD,1,1
        if t>=T_P4:
            if VISIBLE[i]:
                u3=eio3((t-T_P4-0.06*i)/1.0); sx,sy=slot(i)
                cx_=rx+(sx-rx)*u3; cy_=ry+(sy-ry)*u3; D=ROWD+(DSLOT-ROWD)*u3
                al=1-sstep((t-(TB+1.6))/0.5)
            else:
                cx_,cy_,D=rx,ry,ROWD*(1-0.35*sstep((t-T_P4)/0.6))
                al=1-sstep((t-T_P4-0.12*(i-NVIS))/0.6)
        if al>0.01:
            I.blit(cv,avatar_patch(i,D,rp),cx_,cy_,al)
        # landing glow in row
        if 1.0<a<1.7 and t<T_P4:
            I.add_sprite(cv,cv2.resize(I.SPR[40],(300,300)),rx,ry,np.array([255,200,110],np.float32),0.5*(1-(a-1.0)/0.7))
    # ---- coin
    cs=coin_state(t)
    if cs is not None:
        px,py,Dc,al=cs
        if t>T_FLY1 and t<T_P4+0.3:
            I.add_sprite(cv,cv2.resize(I.SPR[40],(520,520)),px,py,np.array([255,190,100],np.float32),(0.26+0.1*math.sin(t*2.0))*sstep((t-T_FLY1)/0.5))
        if al>0.01: I.blit(cv,I.coin_patch(0.0,Dc,None),px,py,al)
    elif patch is not None:
        I.blit(cv,patch,ccx,ccy,appear)
    # ---- flares
    f1=math.exp(-((t-0.9)/0.22)**2)*0.9
    f2=math.exp(-((t-T_SETTLE)/0.28)**2)
    f3=math.exp(-((t-6.45)/0.25)**2)*0.8
    for fv,(fx,fy) in ((f1,(cxs,cys)),(f2,(cxs,cys)),(f3,(FIN[0],FIN[1]))):
        if fv>0.02:
            y0_=int(fy)-40
            M=np.float32([[1,0,fx-W/2],[0,1,0]])
            s_=cv2.warpAffine(I.streak,M,(W,81),flags=cv2.INTER_LINEAR)
            ya=max(0,y0_); yb=min(H,y0_+81)
            cv[ya:yb]+=s_[ya-y0_:yb-y0_,:,None]*np.array([255,215,150],np.float32)*(0.9*fv)
            I.add_sprite(cv,cv2.resize(I.SPR[40],(460,460)),fx,fy,np.array([255,200,110],np.float32),0.45*fv)
    I.draw_shock(cv,t,T_SETTLE,cxs,cys)
    I.draw_sparks(cv,t,6.4,FIN[0],FIN[1])
    for i in range(N):
        if 0<t-(T_M0+DT*i)<1.6 and False: pass
    # ---- text
    I.draw_text(cv,'a1',t,3.2,5.0,540,1450)
    I.draw_text(cv,'a2',t,3.5,5.0,540,1510)
    I.draw_line(cv,t,3.9,5.0,1395,230)
    I.draw_text(cv,'label',t,5.9,T_P4,540,590,fade_out=0.5)
    I.draw_text(cv,'title',t,6.2,T_P4,540,690,dur=1.3,fade_out=0.5)
    I.draw_line(cv,t,6.7,T_P4,790,330)
    for i in range(N):
        ti=T_M0+DT*i
        I.draw_text(cv,f'n{i}',t,ti+0.2,ti+0.9,540,1200,dur=0.7,fade_out=0.22)
        I.draw_text(cv,f'c{i}',t,ti+0.4,ti+0.9,540,1272,dur=0.6,fade_out=0.22)
        if f'r{i}' in I.TXT: I.draw_text(cv,f'r{i}',t,ti+0.5,ti+0.9,540,1336,dur=0.5,fade_out=0.22)
    # glint over banner
    for g0 in [g for g in (TB+4.8,TB+10.3) if g<TEND-1.6]:
        gs=sstep((t-g0)/0.1)*(1-sstep((t-g0-1.5)/0.1))
        if gs>0:
            pos=-300+1900*eio3((t-g0)/1.5)
            band=np.exp(-((xx*0.6+yy*0.45-pos*0.9-300)/90.0)**2).astype(np.float32)
            lum=np.clip(cv.mean(-1)/255.0,0,1)**1.6
            cv+=(band*lum*0.9)[...,None]*np.array([255,230,170],np.float32)*gs
    sh=np.zeros(2)
    if T_SETTLE<=t<T_SETTLE+0.7:
        k=t-T_SETTLE; amp=11*math.exp(-k*7); sh+=(amp*math.sin(k*70),amp*math.cos(k*61))
    if abs(sh).sum()>0.01:
        cv=cv2.warpAffine(cv,np.float32([[1,0,sh[0]],[0,1,sh[1]]]),(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    vig=np.clip(1-(np.hypot((xx-W/2)/(W*0.75),(yy-H/2)/(H*0.62)))**2.2*0.7,0.25,1)[...,None]
    cv*=vig
    cv+=I.grain[int(t*FPS)%6]*0.6
    cv*=1-sstep((t-(TEND-0.5))/0.5)
    return np.clip(cv,0,255).astype(np.uint8)

def run(a,b,name):
    ff=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-preset','medium','-crf','19','-pix_fmt','yuv420p',name],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
    for i in range(a,b):
        ff.stdin.write(render(i/FPS).tobytes())
    ff.stdin.close(); ff.wait()
if __name__=='__main__':
    if sys.argv[1]=='info':
        print(json.dumps(dict(TEND=TEND,N=N,TB=TB,S0=S0,S1=S1,TALL=bool(TALL),FPS=FPS)))
    elif sys.argv[1]=='test':
        for tt in [float(x) for x in sys.argv[2:]]:
            Image.fromarray(render(tt)).save(os.path.join(tempfile.gettempdir(),f'intro_{tt:.2f}.png'))
    else:
        run(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3])
