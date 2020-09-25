import gym
from babyai.levels.verifier import *
from babyai.levels.levelgen import *

from nltk import corpus
import re, string
CHRS = 'abcdefghijklmnopqrstuvwxyz'

# EXP1: EFFICIENCY

class Level_PutNextLocal_d0_e(RoomGridLevel):
    """
    Put an object next to another object, inside a single room
    with no doors, no distractors
    """

    def __init__(self, room_size=8, num_objs=2, seed=None):
        self.num_objs = num_objs
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_objs, all_unique=True)
        self.check_objs_reachable()
        o1, o2 = self._rand_subset(objs, 2)

        self.instrs = PutNextInstr(
            ObjDesc(o1.type, o1.color),
            ObjDesc(o2.type, o2.color)
        )

class Level_PutNextLocal_d2_e(RoomGridLevel):
    """
    Put an object next to another object, inside a single room
    with no doors, two distractors
    """

    def __init__(self, room_size=8, num_objs=4, seed=None):
        self.num_objs = num_objs
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_objs, all_unique=True)
        self.check_objs_reachable()
        o1, o2 = self._rand_subset(objs, 2)

        self.instrs = PutNextInstr(
            ObjDesc(o1.type, o1.color),
            ObjDesc(o2.type, o2.color)
        )

class Level_PutNextLocal_d4_e(RoomGridLevel):
    """
    Put an object next to another object, inside a single room
    with no doors, four distractors
    """

    def __init__(self, room_size=8, num_objs=6, seed=None):
        self.num_objs = num_objs
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors(num_distractors=self.num_objs, all_unique=True)
        self.check_objs_reachable()
        o1, o2 = self._rand_subset(objs, 2)

        self.instrs = PutNextInstr(
            ObjDesc(o1.type, o1.color),
            ObjDesc(o2.type, o2.color)
        )

# EXP2: COMPOSITIONAL

class Level_GoToObj_c(RoomGridLevel):
    """
    Go to an object, inside a single room with no distractors, no doors
    """

    def __init__(self, room_size=8, seed=None, pairs_dict=None, test_mode=None):
        self.pairs_dict = pairs_dict
        self.test_mode = test_mode
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors_train_test(num_instr_objs=1,
                                               num_distractors=0,
                                               pairs_dict=self.pairs_dict,
                                               test_mode=self.test_mode)
        obj = objs[0]
        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_GoToLocal_c(RoomGridLevel):
    """
    Go to an object, inside a single room with no doors, seven distractors
    """

    def __init__(self, room_size=8, seed=None, pairs_dict=None, test_mode=None):
        self.pairs_dict = pairs_dict
        self.test_mode = test_mode
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors_train_test(num_instr_objs=1,
                                               num_distractors=7,
                                               all_unique=False,
                                               pairs_dict=self.pairs_dict,
                                               test_mode=self.test_mode)
        self.check_objs_reachable()
        obj = objs[0]

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


class Level_PutNextLocal_d0_c(RoomGridLevel):
    """
    Put an object next to another object, inside a single room
    with no distractors and no doors
    """

    def __init__(self, room_size=8, seed=None, pairs_dict=None, test_mode=None):
        self.pairs_dict = pairs_dict
        self.test_mode = test_mode
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors_train_test(num_instr_objs=2,
                                               num_distractors=0,
                                               all_unique=True,
                                               pairs_dict=self.pairs_dict,
                                               test_mode=self.test_mode)
        self.check_objs_reachable()
        o1, o2 = objs[0], objs[1]

        self.instrs = PutNextInstr(
            ObjDesc(o1.type, o1.color),
            ObjDesc(o2.type, o2.color)
        )


class Level_PutNextLocal_d2_c(RoomGridLevel):
    """
    Put an object next to another object, inside a single room
    with two distractors, no doors
    """

    def __init__(self, room_size=8, seed=None, pairs_dict=None, test_mode=None):
        self.pairs_dict = pairs_dict
        self.test_mode = test_mode
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors_train_test(num_instr_objs=2,
                                               num_distractors=2,
                                               all_unique=True,
                                               pairs_dict=self.pairs_dict,
                                               test_mode=self.test_mode)
        self.check_objs_reachable()
        o1, o2 = objs[0], objs[1]

        self.instrs = PutNextInstr(
            ObjDesc(o1.type, o1.color),
            ObjDesc(o2.type, o2.color)
        )


class Level_PutNextLocal_d4_c(RoomGridLevel):
    """
    Put an object next to another object, inside a single room
    with four distractors, no doors
    """

    def __init__(self, room_size=8, seed=None, pairs_dict=None, test_mode=None):
        self.pairs_dict = pairs_dict
        self.test_mode = test_mode
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors_train_test(num_instr_objs=2,
                                               num_distractors=4,
                                               all_unique=True,
                                               pairs_dict=self.pairs_dict,
                                               test_mode=self.test_mode)
        self.check_objs_reachable()
        o1, o2 = objs[0], objs[1]

        self.instrs = PutNextInstr(
            ObjDesc(o1.type, o1.color),
            ObjDesc(o2.type, o2.color)
        )


class Level_PutNextLocal_d6_c(RoomGridLevel):
    """
    Put an object next to another object, inside a single room
    with six distractors, no doors
    """

    def __init__(self, room_size=8, seed=None, pairs_dict=None, test_mode=None):
        self.pairs_dict = pairs_dict
        self.test_mode = test_mode
        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent()
        objs = self.add_distractors_train_test(num_instr_objs=2,
                                               num_distractors=6,
                                               all_unique=True,
                                               pairs_dict=self.pairs_dict,
                                               test_mode=self.test_mode)
        self.check_objs_reachable()
        o1, o2 = objs[0], objs[1]

        self.instrs = PutNextInstr(
            ObjDesc(o1.type, o1.color),
            ObjDesc(o2.type, o2.color)
        )


# EXP3: NATURAL LANGUAGE

class Level_PutNextLocal_n(RoomGridLevel):
    """
    Put an object next to another object, inside a single room
    with no doors, but possible distractors
    """

    def __init__(self, room_size=8, num_objs=2, seed=None, num_dists=0,
                 pairs_dict=None, test_instr_mode='babyai', vocab_size=80):
        self.num_objs = num_objs
        self.num_dists = num_dists
        self.pairs_dict = pairs_dict

        self.vocab_size = vocab_size
        self.test_instr_mode = test_instr_mode
        self.test_grammatical_vocab, self.test_random_vocab = self._set_test_vocab()

        super().__init__(
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            seed=seed
        )

    def _set_test_vocab(self):
        # Conservatively set the vocab size because we'll add object properties and colours to the vocab too
        pattern = re.compile("[\d{}]+$".format(re.escape(string.punctuation)))
        test_grammatical_vocab = []

        while len(test_grammatical_vocab) < self.vocab_size:
            word = random.choice(corpus.product_reviews_1.words())
            if len(word) > 3 and not pattern.match(word):
                test_grammatical_vocab.append(word)
        test_random_vocab = [''.join(random.choice(CHRS) for _ in range(random.randint(3, 8)))
                                  for _ in range(self.vocab_size)]
        return test_grammatical_vocab, test_random_vocab

    def gen_mission(self):
        self.place_agent()

        if self.pairs_dict:
            objs = self.add_distractors_train_test(num_instr_objs=self.num_objs,
                                                   num_distractors=self.num_dists,
                                                   all_unique=True,
                                                   pairs_dict=self.pairs_dict,
                                                   test_mode=True)
            self.check_objs_reachable()
            o1, o2 = objs[0], objs[1]
        else:
            objs = self.add_distractors(num_distractors=self.num_objs + self.num_dists, all_unique=True)

            self.check_objs_reachable()
            o1, o2 = self._rand_subset(objs, 2)

        self.instrs = PutNextInstr_n(
            ObjDesc(o1.type, o1.color),
            ObjDesc(o2.type, o2.color),
            test_instr_mode=self.test_instr_mode,
            test_random_vocab = self.test_random_vocab,
            test_grammatical_vocab = self.test_grammatical_vocab
        )


# Register the levels in this file
register_levels(__name__, globals())
