import os, sys

def rename():
    if os.path.exists('./save/checkpoint'):
        # print 'current dir: %s' % os.listdir('./save')
        os.rename('./save/checkpoint', './save/cp_rename')
        print 'rename successfully!'

if __name__ == '__main__':
    rename()