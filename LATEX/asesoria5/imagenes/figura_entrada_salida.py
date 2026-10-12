"""Figura de entrada y salida del preprocesamiento (tesis, Unidad 1).

Dibuja un cubo de 1 m del escaneo antes (.las) y despues (.npy) de
preprocesar.py. Lee el LAS con NumPy (formato de punto 3), sin laspy.

Uso:
    python figura_entrada_salida.py SALIDA.png [CARPETA_DATOS] [NOMBRE]

CARPETA_DATOS debe contener NOMBRE.las, NOMBRE_preprocesado.npy y
NOMBRE_preprocesado.json. Por defecto: Context/DATA y PL7_SCN0001.
"""
import json, struct, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
out=sys.argv[1]
D=(sys.argv[2] if len(sys.argv)>2 else "Context/DATA").rstrip("/")+"/"
NOM=sys.argv[3] if len(sys.argv)>3 else "PL7_SCN0001"
las=D+NOM+".las"
h=open(las,"rb").read(227)
off=struct.unpack_from("<I",h,96)[0]; n=struct.unpack_from("<I",h,107)[0]
scale=np.array(struct.unpack_from("<3d",h,131)); orig=np.array(struct.unpack_from("<3d",h,155))
dt=np.dtype([('X','<i4'),('Y','<i4'),('Z','<i4'),('i','<u2'),('r','u1'),('c','u1'),('a','i1'),('u','u1'),('p','<u2'),('t','<f8'),('R','<u2'),('G','<u2'),('B','<u2')])
raw=np.memmap(las,dtype=dt,mode="r",offset=off,shape=(n,))
c=np.array(json.load(open(D+NOM+"_preprocesado.json"))["offset_utm"])
S=np.load(D+NOM+"_preprocesado.npy")
# elegir un cubo de 1 m con muchos representantes y normales variadas
L=1.0
cell=np.floor(S[:,:3]/L).astype(int)
u,inv,cnt=np.unique(cell,axis=0,return_inverse=True,return_counts=True)
best=None
for k in np.argsort(-cnt)[:400]:
    m=inv==k
    v=S[m,6:9].var(0).sum()
    if best is None or v*np.log(cnt[k])>best[0]: best=(v*np.log(cnt[k]),k)
k=best[1]; lo=u[k]*L; hi=lo+L
# entrada: puntos crudos dentro del mismo cubo
xyz=np.empty((n,3),np.float32)
for j,key in enumerate("XYZ"): xyz[:,j]=(raw[key]*scale[j]+orig[j]-c[j])
m_in=np.all((xyz>=lo)&(xyz<hi),axis=1)
P=xyz[m_in]; C=np.stack([raw['R'][m_in],raw['G'][m_in],raw['B'][m_in]],1)/65535.0
m_out=np.all((S[:,:3]>=lo)&(S[:,:3]<hi),axis=1); Q=S[m_out]
print("cubo",lo,hi,"entrada",len(P),"salida",len(Q))
bright=lambda c: np.clip(c*2.2,0,1)
fig=plt.figure(figsize=(6.6,3.5),dpi=220)
tit=[f"Entrada: {len(P):,} puntos".replace(","," "),f"Salida: {len(Q):,} representantes".replace(","," "),"Salida con sus normales"]
for a in range(2):
    ax=fig.add_subplot(1,2,a+1,projection="3d")
    if a==0: ax.scatter(P[:,0],P[:,1],P[:,2],c=bright(C),s=0.6,linewidths=0)
    else:
        ax.scatter(Q[:,0],Q[:,1],Q[:,2],c=bright(Q[:,3:6]/255.0),s=5,linewidths=0)
        if a==2: ax.quiver(Q[:,0],Q[:,1],Q[:,2],Q[:,6],Q[:,7],Q[:,8],length=0.06,linewidth=0.35,color="tab:red",arrow_length_ratio=0.3)
    ax.set_title(tit[a],fontsize=9)
    ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1]); ax.set_zlim(lo[2],hi[2])
    ax.set_box_aspect((1,1,1)); ax.view_init(elev=18,azim=-60)
    ax.set_xlabel("x (m)",fontsize=6,labelpad=-8); ax.set_ylabel("y (m)",fontsize=6,labelpad=-8); ax.set_zlabel("z (m)",fontsize=6,labelpad=-8)
    ax.tick_params(labelsize=5,pad=-3)
fig.subplots_adjust(left=0.0,right=0.98,bottom=0.04,top=0.93,wspace=0.02)
fig.savefig(out)
