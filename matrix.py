import sys
import struct

def read(file_name, n, m, output_file=sys.stdout):
    n, m = int(n), int(m)
    with open(file_name, 'rb') as stream:
        fmt = struct.Struct('@' + 'd' * (n * m))
        values = fmt.unpack(stream.read(fmt.size))
    output_file.write(''.join('%s\n' % ' '.join(map(str, values[i*m:(i+1)*m])) for i in range(n)))

def write(file_name, input_file=sys.stdin):
    values = (float(each) for line in input_file for each in line.split())
    fmt = struct.Struct('@d')
    with open(file_name, 'w+b') as stream:
        for each in values:
            stream.write(fmt.pack(each))

def main(mode, *args):
    if mode == 'read':
        read(*args)
    elif mode == 'write':
        write(*args)

if __name__ == '__main__':
    main(*sys.argv[1:])
