import os
import shutil
 
source = '/home/sv6234/ECGI/data/TMP_data_GT_new/'
destination = '/home/sv6234/ECGI/data/0.15/'
 
# gather all files
allfiles = os.listdir(source)
 
# iterate on all files to move them to destination folder
for f in allfiles:
    src_path = os.path.join(source, f)
    dst_path = os.path.join(destination, f)
    shutil.move(src_path, dst_path)