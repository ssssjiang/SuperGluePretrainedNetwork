# shu.song@ninebot.com

import torch

# for find hloc
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../..")))

import hloc.extractors as extractors
import hloc.matchers as matchers
from hloc.utils.base_model import dynamic_load
import hloc.extract_features as extract_features
import hloc.match_features as match_features


superpoint_conf = extract_features.confs['superpoint_mower']
Model = dynamic_load(extractors, superpoint_conf['model']['name'])
superpoint = torch.jit.script(Model(superpoint_conf['model']).net.eval())

superglue_conf = match_features.confs['superglue_mower']
Model = dynamic_load(matchers, superglue_conf['model']['name'])
superglue = torch.jit.script(Model(superglue_conf['model']).net.eval())

torch.jit.save(superpoint, "SuperPoint.pt")
torch.jit.save(superglue, "SuperGlue.pt")


