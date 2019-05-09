import sys
import numpy

if len(sys.argv) != 2:
    print('invalid usage: ' + sys.argv[0] + ' <npz-file>')

npzpath = sys.argv[1]

data = numpy.load(npzpath)

def parse(data):
    if isinstance(data, numpy.ndarray):
        return parse(data.tolist())
    elif isinstance(data, numpy.lib.npyio.NpzFile):
        ret = {}
        for k in data.files:
            ret[k] = parse(data[k])
        return ret
    else:
        ty = type(data)
        if ty is dict:
            ret = {}
            for k in data: ret[k] = parse(data[k])
            return ret
        elif ty is int or ty is float:
            return data
        elif ty is str:
            return data
        elif ty is list:
            ret = [];
            for x in data: ret.append(x)
            return ret
        else:
            raise Exception('unknown type: ' + str(type(data)))

def print_json(data):
    w = sys.stdout.write
    w('[')
    directions1 = data.values()
    for n, frame in enumerate(directions1):
        for m, pose in enumerate(frame):
            if n != 0 or m != 0: w(', ')
            w('{"image_id": "' + str(len(directions1) * n + m)+ '", "category_id": 1, ')
            joinedpts = []
            for pt in pose: joinedpts +=  [pt[0] * 750, pt[1] * 750, pt[2] * 750]
            w('"keypoints": [' + ', '.join(map(lambda x: str(x), joinedpts)) + ']')
            w('}')
    w(']')

print_json(parse(data))
