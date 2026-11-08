"""Inspección de solo lectura de DATA; no ejecuta ni repite el preprocesamiento.

Usa NumPy y la biblioteca estándar. LAS: decodificador de formato 3.
PLY: propiedades escalares de vértices en binario little-endian.
"""
from pathlib import Path
import csv
import json
import struct
import zipfile
import zlib
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'DATA'

def bounds(a):
    return {'min': np.min(a, axis=0).tolist(), 'max': np.max(a, axis=0).tolist()}

def main():
    out = {}
    las = DATA / 'PL7_SCN0001.las'
    with las.open('rb') as f:
        h = f.read(227)
        unpack = lambda fmt, pos: struct.unpack_from('<' + fmt, h, pos)
        header_size, offset, vlr_n = unpack('HII', 94)
        fmt = h[104]
        recsize = unpack('H', 105)[0]
        n = unpack('I', 107)[0]
        scale = np.array(unpack('3d', 131))
        origin = np.array(unpack('3d', 155))
        hdr = {
            'signature': h[:4].decode(), 'version': f'{h[24]}.{h[25]}',
            'system_identifier': h[26:58].rstrip(b'\0').decode(errors='replace'),
            'generating_software': h[58:90].rstrip(b'\0').decode(errors='replace'),
            'creation_day_year': unpack('HH', 90),
            'global_encoding': unpack('H', 6)[0], 'header_size': header_size,
            'point_offset': offset, 'vlr_count': vlr_n,
            'point_format': fmt, 'record_bytes': recsize, 'point_count': n,
            'points_by_return': unpack('5I', 111),
            'scales': scale.tolist(), 'offsets': origin.tolist(),
            'header_bounds_maxmin_xyz': unpack('6d', 179), 'vlrs': [],
        }
        f.seek(header_size)
        for _ in range(vlr_n):
            vh = f.read(54)
            _, user, rid, length, description = struct.unpack('<H16sHH32s', vh)
            data = f.read(length)
            v = {'user_id': user.rstrip(b'\0').decode(errors='replace'),
                 'record_id': rid, 'length': length,
                 'description': description.rstrip(b'\0').decode(errors='replace'),
                 'hex': data.hex()}
            if rid == 34735 and len(data) % 2 == 0:
                v['geokey_shorts'] = struct.unpack('<' + 'H'*(len(data)//2), data)
            if rid in (34737, 2112):
                v['text'] = data.decode(errors='replace')
            hdr['vlrs'].append(v)
    assert fmt == 3 and recsize == 34, (fmt, recsize)
    dt = np.dtype([('X','<i4'),('Y','<i4'),('Z','<i4'),('intensity','<u2'),
                   ('return_flags','u1'),('classification_flags','u1'),
                   ('scan_angle_rank','i1'),('user_data','u1'),
                   ('point_source_id','<u2'),('gps_time','<f8'),
                   ('red','<u2'),('green','<u2'),('blue','<u2')])
    raw = np.memmap(las, dtype=dt, mode='r', offset=offset, shape=(n,))
    hdr['size_expected'] = offset+n*recsize
    hdr['size_actual'] = las.stat().st_size
    out['las_header'] = hdr
    fields = {k: raw[k] for k in dt.names if k not in ('return_flags','classification_flags')}
    for name, shift, mask in [('return_number',0,7),('number_of_returns',3,7),
                              ('scan_direction_flag',6,1),('edge_of_flight_line',7,1)]:
        fields[name] = (raw['return_flags'] >> shift) & mask
    for name, shift, mask in [('classification',0,31),('synthetic',5,1),
                              ('key_point',6,1),('withheld',7,1)]:
        fields[name] = (raw['classification_flags'] >> shift) & mask
    out['las_fields'] = {k: {'dtype':str(v.dtype), **bounds(v)} for k,v in fields.items()}
    out['gps_time_equals_record_index'] = all(
        np.array_equal(raw['gps_time'][i:i+1_000_000], np.arange(i,min(i+1_000_000,n)))
        for i in range(0,n,1_000_000))
    out['rgb_all_multiples_of_256'] = all(bool(np.all(raw[k] % 256 == 0)) for k in ('red','green','blue'))
    for k in ['classification','return_number','number_of_returns','scan_direction_flag',
              'edge_of_flight_line','synthetic','key_point','withheld','user_data','point_source_id']:
        vals, counts = np.unique(fields[k], return_counts=True)
        out['las_fields'][k]['counts'] = dict(zip(map(str, vals.tolist()), counts.tolist()))
    xyz_bounds = {'min':[], 'max':[]}
    for i, key in enumerate('XYZ'):
        xyz_bounds['min'].append(float(raw[key].min()*scale[i]+origin[i]))
        xyz_bounds['max'].append(float(raw[key].max()*scale[i]+origin[i]))
    out['las_xyz_scaled_bounds'] = xyz_bounds
    a = np.load(DATA/'PL7_SCN0001_preprocesado.npy', mmap_mode='r')
    out['npy'] = {'shape':a.shape, 'dtype':str(a.dtype), 'first10':a[:10].tolist(),
                  **bounds(a), 'finite':bool(np.isfinite(a).all()),
                  'normal_length':bounds(np.linalg.norm(a[:,6:9],axis=1))}
    # Reproduce only CenterShift + grid indices + FNV hashing in NumPy.
    # No neural model is loaded or run here.
    coord = np.array(a[:,:3],copy=True)
    lo, hi = coord.min(0), coord.max(0)
    shift = [(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, lo[2]]
    coord -= shift
    grid = np.floor(coord / np.array(0.02)).astype(int)
    grid -= grid.min(0)
    hashes = np.full(len(grid),np.uint64(14695981039346656037),dtype=np.uint64)
    for j in range(3):
        hashes *= np.uint64(1099511628211)
        hashes ^= grid[:,j].astype(np.uint64)
    out['sonata_grid_default_numpy_reproduction'] = {
        'grid_size':0.02, 'center_shift':list(map(float,shift)),
        'count_unique_hashes':len(np.unique(hashes)),
        'count_unique_grid_coordinates':len(np.unique(grid,axis=0)),
        'model_executed':False}
    meta = json.loads((DATA/'PL7_SCN0001_preprocesado.json').read_text())
    out['metadata'] = meta
    boxcenter = (np.array(xyz_bounds['min'])+xyz_bounds['max'])/2
    out['offset_matches_bbox_center'] = bool(np.allclose(boxcenter, meta['offset_utm'], rtol=0, atol=1e-8))
    for filename, arr in [('PL7_muestra_100.csv',None),('PL7_preprocesado_muestra_100.csv',a)]:
        with (DATA/filename).open(newline='') as f:
            reader = csv.reader(f)
            columns = next(reader)
            rows = list(reader)
        info = {'columns':columns, 'row_count':len(rows), 'column_count':len(columns), 'first10':rows[:10]}
        indices = np.linspace(0, (n if arr is None else len(arr))-1, len(rows), dtype=int)
        samples = np.array(rows,dtype=float)
        if arr is None:
            rr = raw[indices]
            actual = np.column_stack([rr[k]*scale[i]+origin[i] for i,k in enumerate('XYZ')]+
                      [fields[k][indices] for k in columns[3:]])
        else:
            actual = arr[indices]
        info['linspace_indices'] = indices.tolist()
        info['sample_max_abs_error_per_column'] = np.abs(actual-samples).max(axis=0).tolist()
        info['matches_linspace_with_csv_rounding'] = bool(np.allclose(actual,samples,rtol=0,atol=0.000051))
        out[filename] = info
    ply = DATA/'PL7_SCN0001_preprocesado.ply'
    types = {'double':'<f8','float':'<f4','uchar':'u1','uint8':'u1','int':'<i4'}
    props = []
    with ply.open('rb') as f:
        lines=[]
        while True:
            s=f.readline().decode('ascii').rstrip()
            lines.append(s)
            if s.startswith('format '): plyformat=s.split()[1]
            if s.startswith('element vertex '): pn=int(s.split()[-1])
            if s.startswith('property '):
                _,t,name=s.split(); props.append((name,types[t]))
            if s=='end_header': break
        po=f.tell()
    assert plyformat=='binary_little_endian'
    p=np.memmap(ply, dtype=np.dtype(props), mode='r', offset=po, shape=(pn,))
    out['ply']={'header':lines,'count':pn,'properties':props,'size_expected':po+pn*p.dtype.itemsize,
                'size_actual':ply.stat().st_size,'comparison':{}}
    for name, col in [('x',0),('y',1),('z',2),('red',3),('green',4),('blue',5),('nx',6),('ny',7),('nz',8)]:
        out['ply']['comparison'][name]={'max_abs_difference_npy':float(np.max(np.abs(p[name]-a[:,col]))),
             'equal_after_float32':bool(np.array_equal(p[name].astype(np.float32),a[:,col]))}
        if name in ('red','green','blue'):
            out['ply']['comparison'][name]['equals_rounded_npy']=bool(np.array_equal(p[name], np.rint(a[:,col]).clip(0,255).astype(np.uint8)))
    comparisons=[]
    for i in range(5):
        r = raw[out['PL7_muestra_100.csv']['linspace_indices'][i]]
        orig = np.array([r[k]*scale[j]+origin[j] for j,k in enumerate('XYZ')])
        centered=orig-meta['offset_utm']
        dist2=np.sum((a[:,:3].astype(np.float64)-centered)**2,axis=1)
        ni=int(dist2.argmin())
        comparisons.append({'sample_row_1based':i+1,'original_index_0based':out['PL7_muestra_100.csv']['linspace_indices'][i],
              'original_xyz':orig.tolist(), 'original_rgb':[int(r[k]) for k in ('red','green','blue')],
              'centered_xyz':centered.tolist(), 'rgb_div256':[float(r[k]/256) for k in ('red','green','blue')],
              'nearest_npy_index':ni,'distance_m':float(np.sqrt(dist2[ni])), 'nearest_npy_row':a[ni].tolist(),
              'warning':'Nearest spatial point is not confirmed to be the same point or its voxel representative.'})
    out['five_nearest_comparisons']=comparisons
    out['zip']=[]
    with zipfile.ZipFile(ROOT/'DATA.zip') as z:
        for member in z.infolist():
            if member.is_dir(): continue
            file=ROOT/member.filename
            crc=0
            with file.open('rb') as f:
                for block in iter(lambda:f.read(4*1024*1024),b''):
                    crc=zlib.crc32(block,crc)
            out['zip'].append({'member':member.filename,'size':member.file_size,
                               'matches_local_size_crc':file.stat().st_size==member.file_size and crc==member.CRC})
    dest=ROOT/'inspeccion_DATA.json'
    dest.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(dest)
    print(json.dumps({k:out[k] for k in ['las_header','las_fields','las_xyz_scaled_bounds','npy','ply','five_nearest_comparisons','zip']},indent=2))

if __name__=='__main__':
    main()
