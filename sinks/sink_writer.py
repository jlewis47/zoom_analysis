import numpy as np
from f90_tools.IO import write_record
from gremlin.read_sim_params import ramses_sim
from zoom_analysis.sinks.sink_reader import get_coarse_dts, snap_to_coarse_step


def write_sink_bin(fname, data, **kwargs):



    hagn = False
    if "hagn" in kwargs.keys():
        hagn = kwargs["hagn"]
    hdr_only = kwargs.get("hdr_only", False)
    
    tgt_fields = kwargs.get("tgt_fields", None)
    

    fields = [
        ("identity", 1, "i4"),
        ("mass", 1, "f8"),
        ("position", 3, "f8"),
        ("velocity", 3, "f8"),
        ("angular_mom", 3, "f8"),
        ("birth_time", 1, "f8"),
        ("dMBH_coarse", 1, "f8"),
        ("dMEd_coarse", 1, "f8"),
        ("dMsmbh", 1, "f8"),
        ("dens", 1, "f8"),
        ("csound", 1, "f8"),
        ("vrel", 1, "f8"),
        ("Esave", 1, "f8"),
        ("j_sink", 3, "f8"),
        ("spin_magnitude", 1, "f8"),
        ("eps_sink", 1, "f8"),
    ]  # what is sink stat ? ndim*2+1... loop on levels
    # if RT... LAGN_coarse

    # print(hagn)

    if hagn:  # accomodate hagn format

        fields = (np.asarray(fields)[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]]).tolist()
        # hagn doesnt have birth time or spin stuff
        fields = [
            (f[0], int(f[1]), f[2]) for f in fields
        ]  # need to remake tuples avec this horrible trick, integers where strings

    if tgt_fields is None:
        tgt_fields = fields

    with open(fname, "wb") as dest:

        write_record(dest, data["nsink"], "i4")
        write_record(dest, data["ndim"], "i4")
        write_record(dest, data["aexp"], "f8")
        write_record(dest, data["unit_l"], "f8")
        write_record(dest, data["unit_d"], "f8")
        write_record(dest, data["unit_t"], "f8")

        for field in tgt_fields:

            print(field[0],np.shape(data[field[0]]),field[2])

            # write_record(dest, data[field[0]], field[2], data[field[0]].min(), data[field[0]].max())
            dat = data[field[0]]
            sh = dat.shape
            if len(sh)==1:
                write_record(dest, dat, dtypes=field[2])
            elif len(sh)==2:
                short_shape = min(sh)
                for idim in range(short_shape):
                    write_record(dest, dat[:,idim], dtypes=field[2])



    return 1



def write_sink_output(fname, fdesc, data, **kwargs):



    hagn = False
    if "hagn" in kwargs.keys():
        hagn = kwargs["hagn"]
    hdr_only = kwargs.get("hdr_only", False)
    tgt_fields = kwargs.get("tgt_fields", None)
    

    desc = np.genfromtxt(fdesc,comments='#',delimiter=',',dtype=np.dtype(('i4,U15,U2')))

    fields=[]

    type_dict={
        'i':'i4',
        'd':'f8'
    }

    for field in desc:

        fields.append((field[1].strip(),1,type_dict[field[2].strip()]))
        print(fields[-1])
    # print(hagn)

    if tgt_fields is None:
        tgt_fields = fields
        

    # print(desc,fields)

    with open(fname, "wb") as dest:

        write_record(dest, data["nsink"], "i4")
        write_record(dest, data["nindsink"], "i4")

        for field in tgt_fields:

            print(field[0],np.shape(data[field[0]]),field[2])

            # write_record(dest, data[field[0]], field[2], data[field[0]].min(), data[field[0]].max())
            dat = data[field[0]]
            sh = dat.shape
            if len(sh)==1:
                write_record(dest, dat, dtypes=field[2])
            elif len(sh)==2:
                short_shape = min(sh)
                for idim in range(short_shape):
                    write_record(dest, dat[:,idim], dtypes=field[2])



    return 1

