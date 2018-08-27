import sys, os
from glob import glob

def pipe(cmd):
    fp = os.popen(cmd)
    res = fp.read()
    stat = fp.close()
    return res, stat
 
def main(script, files='*.py'):
    for file in glob(files):
        if file in ['loop.py']:
            continue
            
        #cmd = '2to3 -w -f print %s' % (file,) 
        cmd = "sed -i 's/Gaussian/Normal/g' %s" % (file,) 
        print cmd
        
        res, stat = pipe(cmd)
        print res, stat
  

if __name__ == '__main__':
    main(*sys.argv)
