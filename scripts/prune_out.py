import os
import glob


for dir in glob.glob("../out/*/*"):
    tbs = []
    for tb in glob.glob(dir + "/events.*"):
        tbs.append((tb, os.path.getsize(tb)))
    tbs.sort(key=lambda x: x[1], reverse=True)
    if len(tbs) > 1 and tbs[0][1] > tbs[1][1]:
        for bad_tb in tbs[1:]:
            os.rmdir(bad_tb[0])
