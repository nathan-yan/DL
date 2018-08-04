import os
import sys

def findFiles(path):
    files = []
    for dirpath, _, files_ in os.walk(path):
        files += [os.path.join(dirpath, f) for f in files_ if f[-2:] == '.h' or f[-2] == '.c']
    
    return files

def createDataset(path = '../datasets/linux', write_path = '../datasets/linux_concat.txt'):
    files = findFiles(path)
    print(len(files))
    longest_name = 0
    for f in files:
        if len(f) > longest_name:
            longest_name = len(f)

    # Create a file to write to
    with open(write_path, 'w') as data:
        c = 0
        for filename in files:
            c += 1
            f = open(filename, 'r')

            string = "Writing %s to %s" % (filename, path)
            space = ' ' * (longest_name - len(filename) - 12 + 1)

            print("\r%s %s %s%%" % (string, space, c / len(files)) , end = '')    

            sys.stdout.flush()

            try:
                data.write(f.read() + "\n")
            except UnicodeDecodeError:
                f.close()

            f.close()

if __name__ == "__main__":
    createDataset()