from skimage.color import rgb2gray
import imageio
from zipfile import ZipFile
import sys
import os
import numpy as np
from multiprocessing import Pool, cpu_count

def dr_cropped_dims(img, threshold_of_max=.1,axis=-1):
    if axis < 0:
        axis = 0
    flat = rgb2gray(img)
    
    ontox = flat.sum(axis=axis)
    mask = np.where(ontox > .1*np.max(ontox))
    not_mask = np.where(ontox <= .1*np.max(ontox))
    
    ontox[ mask ] = 1
    ontox[ not_mask ] = 0
    
    lox = np.argmax(ontox)
    hix = ontox.shape[0] - np.argmax(ontox[::-1])
    return lox,hix
    
def dr_crop(img, threshold_of_max=.1):
    """assumes a RGB image to come in, crops image to threshold of max"""
    flat = rgb2gray(img)
    # x first
    lox, hix = dr_cropped_dims(img, threshold_of_max,axis=0)
    
    # y second
    loy, hiy = dr_cropped_dims(img, threshold_of_max,axis=1)
    
    return img[loy:hiy,lox:hix,:]

def crop_uri_shape(uri):
    img = imageio.imread(uri)
    cropped = dr_crop(img)
    line = "{},{},{}".format(",".join([ str(it) for it in img.shape ]),
                             ",".join([ str(it) for it in cropped.shape]),
                             uri)
    return line
    
if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print("usage: python3 cropped_size.py <file> (<files> ...)")
        print("       produces a csv file with shape[0],shape[1],cropped[0],cropped[1],name")
        sys.exit(1)
        
    outcsv = open("output.csv","w")
    workers = Pool(processes=cpu_count())
    
    for it in sys.argv[1:]:
        if not os.path.isfile(it):
            print(it,"is not a file, skipping it")
            continue
            
        if "zip" in os.path.splitext(it)[-1].lower():
            entries = []
            with ZipFile(it) as archive:
                iid = 0
                entries = [ os.path.join(it,itz.filename) for itz in archive.infolist() if not itz.is_dir()]
            print(f"{it} is a zipfile with {len(entries)} entries")
            lines = workers.map(crop_uri_shape,entries)
            outcsv.write("\n".join(lines))
            outcsv.write("\n")
        else:
            img = imageio.imread(it)
            cropped = dr_crop(img)
            line = "{},{},{}".format(",".join(img.shape),",".join(cropped.shape),os.path.split(it)[-1])
            outcsv.write(line+"\n")
    outcsv.close()
    print("wrote output.csv")