import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sweep import prep, levels
from PIL import Image
CB=4.5
Hc,Hcf=prep(CB)
X=np.linspace(1.0,3.5,126)
L=[levels(Hc,Hcf,x) for x in X]
T2=[l['T2'] for l in L]; T1F=[l['T1F'] for l in L]; T1P=[l['T1P'] for l in L]; E=[l['E'] for l in L]
# crossing
f=lambda dq: levels(Hc,Hcf,dq)['T2']-levels(Hc,Hcf,dq)['E']
a,b=1.8,2.4
for _ in range(40):
    m=(a+b)/2; a,b=(a,m) if f(a)*f(m)<=0 else (m,b)
xc=(a+b)/2; yc=10*xc
print('crossing',xc)
plt.rcParams.update({'font.family':'Liberation Serif','font.size':8.0,'mathtext.fontset':'stix'})
BL,GR,OR,GY='#0072BD','#77AC30','#D95319','#8C8C8C'
fig=plt.figure(figsize=(1800/600,1646/600),dpi=600)
ax=fig.add_axes([310/1800,1-1398/1646,(1552-310)/1800,(1398-127)/1646])
ax.axvspan(2.23,2.42,color='#E8D8DE',zorder=0)
for v in (2.23,2.42): ax.axvline(v,color='#9a9a9a',lw=0.6,ls=':',zorder=1)
ax.plot(X,T1P,color=GY,lw=1.1); ax.plot(X,T1F,color=GY,lw=1.1)
ax.plot(X,E,color=GR,lw=1.6); ax.plot(X,T2,color=BL,lw=1.6)
ax.axhline(0,color='k',lw=1.4)
ax.plot([xc],[yc],'o',ms=4.2,mfc='white',mec='k',mew=1.0,zorder=6)
ax.annotate(f'{xc:.2f}',xy=(xc,yc),xytext=(xc+0.12,yc-7.5),fontsize=7.6,
            arrowprops=dict(arrowstyle='-',lw=0.6,color='k'),zorder=7)
for x,c in [(1.85,BL),(2.3,OR),(2.8,GR)]:
    y=10*x
    ax.plot([x,x],[y,0.6],ls=':',color=c,lw=1.1)
    ax.plot([x],[0.1],marker='v',color=c,ms=5.5,clip_on=False,zorder=8)
ax.set_xlim(1,3.5); ax.set_ylim(-1,45)
ax.set_xlabel('$Dq/B$'); ax.set_ylabel('$E/B$')
ax.tick_params(direction='in',top=True,right=True,length=3,width=0.7)
ax.set_xticks([1,1.5,2,2.5,3,3.5]); ax.set_yticks([0,10,20,30,40])
ax.minorticks_on(); ax.tick_params(which='minor',direction='in',top=True,right=True,length=1.6,width=0.5)
ax.text(1.1,38.5,r'$^4$T$_1$(P)',color=GY); ax.text(1.4,25.5,r'$^4$T$_1$(F)',color=GY)
ax.text(1.17,13.7,r'$^4$T$_2$',color=BL); ax.text(3.2,22.6,r'$^2$E',color=GR)
ax.text(1.12,1.6,r'$^4$A$_2$',color='k')
ax.text(3.42,3.0,f'C/B = {CB:g}',fontsize=7.2,color='#444444',ha='right')
for x,t in [(1.6,'WCF'),(2.325,'ICF'),(2.88,'SCF')]:
    ax.text(x,46.2,t,ha='center',va='bottom',color='#555555',fontsize=7.4)
fig.text(0.02,1-180/1646,'(a)',fontweight='bold',fontsize=8.6,va='center')
for s in ax.spines.values(): s.set_linewidth(0.8)
fig.savefig('fig1a.png',dpi=600)
A=Image.open('fig1a.png').convert('RGB')
import sys
# panel (b) is taken from the previous version of Figure 1; pass its path as the first argument
prev = sys.argv[1] if len(sys.argv) > 1 else 'Figure_1_crystal_field_regimes.png'
O=Image.open(prev).convert('RGB')
out=O.copy(); out.paste(A.crop((0,0,1800,1646)),(0,0))
out.save('Figure1_CB45.png',dpi=(600,600))
