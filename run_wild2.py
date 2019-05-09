import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import json
from functools import reduce
from collections import OrderedDict

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random

args = parse_args()

def load_keypoints(filepath, gen=False, scaler=1.0):
    def groupby(key, fs):
        def join(xs):
            def append(xs, x):
                for k, v in x.items():
                    if k in xs: xs[k].append(v)
                    else: xs[k] = [v]
                return xs
            return reduce(lambda acc, b: append(acc, b), xs, {})
        res = OrderedDict()
        for f in fs:
            f2 = f.copy()
            kval = str(f[key])
            f2.pop(key, None)
            if kval in res: res[kval].append(f2)
            else: res[kval] = [f2]
        return [(k, join(vs)) for (k, vs) in res.items()]
    def pt_add(a, b):
        return [a[0]+b[0], a[1]+b[1], a[2]*b[2]]
    def pt_scl(a, s):
        return [a[0]*s, a[1]*s, a[2]*s]
    def pt_interp(a, b, w):
        return [a[0]+w*(b[0]-a[0]), a[1]+w*(b[1]-a[1]), a[2]*(1-w)+b[2]*w]
    def remap(keys):
        mapping = [11,14,12,15,13,16,4,1,5,2,6,3,9,0,7,8,10]
        ret = [0] * 17
        for i, v in enumerate(keys):
            ret[mapping[i]] = v
        return ret
    def gen_missing_keypoints(keys):
        assert len(keys) == 17
        head = pt_scl(pt_add(keys[3], keys[4]), .5)
        keys2 = [i for j, i in enumerate(keys) if j not in [0,1,2,3,4]]
        keys2.append(head)
        root = pt_scl(pt_add(keys2[6], keys2[7]), .5)
        keys2.append(root)
        thorax = pt_scl(pt_add(keys2[0], keys2[1]), .5)
        keys2.append(thorax)
        neck = pt_interp(thorax, head, .25)
        keys2.append(neck)
        keys2.append(head)
        return keys2
    def filter_max(frame):
        if len(frame['keypoints']) == 1:
            return frame['keypoints'][0]
        else:
            return frame['keypoints'][frame['score'].index(max(frame['score']))]
    def parsePoints(xs): return [pt_scl(pt, scaler) for pt in zip(xs[0::3], xs[1::3], xs[2::3])]
    with open(filepath, 'r') as f:
        return [parsePoints(nums) for nums in
        #return [remap(gen_missing_keypoints(parsePoints(nums))) for nums in
                [filter_max(frame[1]) for frame in
                    groupby('image_id',
                        json.loads(f.read()))]]

try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

print('Loading 2D detections...', args.input)
#poses2d = np.load(args.input)['positions_2d'].item()['S1']['Directions 1'][0]
poses2d = np.array(load_keypoints('../results/maya/baseball.json', gen=False, scaler=1))

# Normalize camera frame
poses2d[..., :2] = normalize_screen_coordinates(poses2d[..., :2], w=1080, h=1080) #(591, 17, 3) => (591, 17, 2)


model_pos = TemporalModel(
        poses2d.shape[-2], # num_joints_in 17
        poses2d.shape[-1], # in_features 3
        17,                # num_joints_out
        filter_widths=[int(x) for x in args.architecture.split(',')],
        causal=args.causal,
        dropout=args.dropout,
        channels=args.channels,
        dense=args.dense)

receptive_field = model_pos.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2 # Padding on each side
causal_shift = pad if args.causal else 0

model_params = reduce(lambda a, p: a + p.numel(), model_pos.parameters(), 0)
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
    model_pos = model_pos.cuda()
 
chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
print('Loading checkpoint', chk_filename)
checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
print('This model was trained for {} epochs'.format(checkpoint['epoch']))
model_pos.load_state_dict(checkpoint['model_pos'])
    

def evaluate(test_generator):
    with torch.no_grad():
        model_pos.eval()
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            predicted_3d_pos = model_pos(inputs_2d)
                
            return predicted_3d_pos.squeeze(0).cpu().numpy()


gen = UnchunkedGenerator(None, None, [poses2d.copy()], pad=pad, causal_shift=causal_shift)
prediction = evaluate(gen)

prediction = camera_to_world(prediction, R=np.array([0, 0, 0, 0], dtype=np.float32), t=0)
# We don't have the trajectory, but at least we can rebase the height
prediction[:, :, 2] -= np.min(prediction[:, :, 2])

np.savez(args.output, prediction)
