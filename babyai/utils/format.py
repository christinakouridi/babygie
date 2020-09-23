import os
import json
import numpy
import re
import torch
import babyai.rl

from .. import utils
from .. import gie


def get_vocab_path(model_name):
    return os.path.join(utils.get_model_dir(model_name), "vocab.json")


class Vocabulary:
    def __init__(self, model_name):
        self.path = get_vocab_path(model_name)
        self.max_size = 100
        if os.path.exists(self.path):
            self.vocab = json.load(open(self.path))
        else:
            self.vocab = {}

    def __getitem__(self, token):
        if not (token in self.vocab.keys()):
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

    def save(self, path=None):
        if path is None:
            path = self.path
        utils.create_folders_if_necessary(path)
        json.dump(self.vocab, open(path, "w"))

    def copy_vocab_from(self, other):
        '''
        Copy the vocabulary of another Vocabulary object to the current object.
        '''
        self.vocab.update(other.vocab)


class InstructionsPreprocessor(object):
    def __init__(self, model_name, load_vocab_from=None):
        self.model_name = model_name
        self.vocab = Vocabulary(model_name)

        path = get_vocab_path(model_name)
        if not os.path.exists(path) and load_vocab_from is not None:
            # self.vocab.vocab should be an empty dict
            secondary_path = get_vocab_path(load_vocab_from)
            if os.path.exists(secondary_path):
                old_vocab = Vocabulary(load_vocab_from)
                self.vocab.copy_vocab_from(old_vocab)
            else:
                raise FileNotFoundError('No pre-trained model under the specified name')

    def __call__(self, obss, device=None):
        raw_instrs = []
        max_instr_len = 0

        for obs in obss:
            tokens = re.findall("([a-z]+)", obs["mission"].lower())
            # This is where the Vocab object is called, and new words are inserted where necessary
            instr = numpy.array([self.vocab[token] for token in tokens])
            raw_instrs.append(instr)
            max_instr_len = max(len(instr), max_instr_len)

        instrs = numpy.zeros((len(obss), max_instr_len))

        for i, instr in enumerate(raw_instrs):
            instrs[i, :len(instr)] = instr

        instrs = torch.tensor(instrs, device=device, dtype=torch.long)
        return instrs


class RawImagePreprocessor(object):
    def __call__(self, obss, device=None):
        images = numpy.array([obs["image"] for obs in obss])
        images = torch.tensor(images, device=device, dtype=torch.float)
        return images


class IntImagePreprocessor(object):
    def __init__(self, num_channels, max_high=255):
        self.num_channels = num_channels
        self.max_high = max_high
        self.offsets = numpy.arange(num_channels) * max_high
        self.max_size = int(num_channels * max_high)

    def __call__(self, obss, device=None):
        images = numpy.array([obs["image"] for obs in obss])
        # The padding index is 0 for all the channels
        images = (images + self.offsets) * (images > 0)
        images = torch.tensor(images, device=device, dtype=torch.long)
        return images


class GieInstructionsPreprocessor(object):
    def __init__(self, model_name):
        self.model_name = model_name
        self.vocab = Vocabulary(model_name)
        self.parser = gie.instruction_parser.Parser()
        self.parser_history = {}

    @staticmethod
    def add_self_loops(edges_idx, num_nodes=None):
        if not num_nodes:
            num_nodes = max(edges_idx)
        for i in range(num_nodes):
            edges_idx += [(i, i)]
        return edges_idx

    def __call__(self, obss, device=None):
        edges_indices = []
        words_indices = []
        depths = []
        missions_tokens = []

        for obs in obss:
            mission = obs['mission'].lower()
            mission_tokens = tuple(mission.split())

            if mission in self.parser_history.keys():
                local_edges_idx, global_words_idx, depth = self.parser_history.get(mission)
            else:
                local_edges_idx, _, depth = self.parser.parse(mission)
                
                global_words_idx = [self.vocab[word] for word in mission_tokens]

                num_nodes = len(mission_tokens)
                local_edges_idx = self.add_self_loops(local_edges_idx, num_nodes)

                self.parser_history[mission] = (local_edges_idx, global_words_idx, depth)

            edges_indices.append(torch.tensor(local_edges_idx, dtype=torch.long, device=device))
            words_indices.append(torch.tensor(global_words_idx, dtype=torch.long, device=device))
            missions_tokens.append(mission_tokens)
            depths.append([depth])

        edges_indices_t = torch.stack(edges_indices).to(device)
        words_indices_t = torch.stack(words_indices).to(device)

        depths_t = torch.tensor(depths, dtype=torch.long, device=device)

        return edges_indices_t, words_indices_t, depths_t, numpy.array(missions_tokens)

class ObssPreprocessor:
    def __init__(self, *, instr_arch: str, model_name: str, obs_space = None, load_vocab_from = None):
        self.image_preproc = RawImagePreprocessor()
        # self.instr_preproc = InstructionsPreprocessor(model_name, load_vocab_from)

        if 'gie' in instr_arch or 'bert' in instr_arch:
            self.instr_preproc = GieInstructionsPreprocessor(model_name)
        else:
            self.instr_preproc = InstructionsPreprocessor(model_name, load_vocab_from)

        self.vocab = self.instr_preproc.vocab
        self.obs_space = {
            "image": 147,
            "instr": self.vocab.max_size
        }

    def __call__(self, obss, device=None):
        obs_ = babyai.rl.DictList()

        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        if "instr" in self.obs_space.keys():
            obs_.instr = self.instr_preproc(obss, device=device)

        return obs_


class IntObssPreprocessor(object):
    def __init__(self, model_name, obs_space, load_vocab_from=None):
        image_obs_space = obs_space.spaces["image"]
        self.image_preproc = IntImagePreprocessor(image_obs_space.shape[-1],
                                                  max_high=image_obs_space.high.max())
        self.instr_preproc = InstructionsPreprocessor(load_vocab_from or model_name)
        self.vocab = self.instr_preproc.vocab
        self.obs_space = {
            "image": self.image_preproc.max_size,
            "instr": self.vocab.max_size
        }

    def __call__(self, obss, device=None):
        obs_ = babyai.rl.DictList()

        if "image" in self.obs_space.keys():
            obs_.image = self.image_preproc(obss, device=device)

        if "instr" in self.obs_space.keys():
            obs_.instr = self.instr_preproc(obss, device=device)

        return obs_
