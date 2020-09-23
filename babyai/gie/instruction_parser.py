# Parser class to predict syntax tree for an instruction based on AllenNLP's dependency parser.
from collections import deque
from typing import List, Tuple, Dict

from allennlp.predictors.predictor import Predictor

_ALLEN_NLP_DEP_PARSER_PATH = "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz"


class Parser:
    """
    Predicts a syntax tree for an instruction using AllenNLP's pre-trained dependency parser.
    For a demo, visit https://demo.allennlp.org/dependency-parsing
    """

    def __init__(self, parser_type: str = 'dep'):

        if parser_type != 'dep':
            raise AssertionError("Expected parser type of 'dep'. More types will be added in future.")

        self.parser_type = parser_type
        self.predictor = Predictor.from_path(_ALLEN_NLP_DEP_PARSER_PATH)

    @staticmethod
    def _format_tree(tree: Dict[str, List], word_indices: Dict[int, int]) -> Tuple[List[Tuple[str, str]],
                                                                                   List[Tuple[int, int]], int]:
        """
        BFS traversal through tree to create a list of edges.
        """
        depth = 0 # initialise depth
        edges_wrd, edges_idx = [], []
        stack = deque([[tree, depth]])

        while stack or tree.get('children'):
            tree, depth = stack.popleft()
            if tree.get('children'):
                depth += 1
                for child in tree['children']:
                    edges_wrd.append((child['word'], tree['word']))
                    edges_idx.append((word_indices[child['spans'][0]['end']],
                                      word_indices[tree['spans'][0]['end']]))
                    stack.append([child, depth])
        return edges_wrd, edges_idx, depth

    @staticmethod
    def _get_word_indices(sentence: str) -> Dict[int, int]:
        """
        Creates a dict that enables mapping a node in the parsed tree to a word
        index, in order to create a list of edges with word indices. The parser
        provides the start (inclusive) and end (non-inclusive) character indices for
        each word representing a node. Note that no special characters are dropped
        """
        word_end_bounds = []  # init [0] to get start indices instead
        for char in range(len(sentence)):
            if sentence[char] == ' ':
                word_end_bounds.append(char + 1)
        word_indices = {b: i for i, b in enumerate(word_end_bounds)}
        return word_indices

    def parse(self, instr_lang: str) -> Tuple[List[Tuple[int, int]], Dict[int, str], int]:
        """
        Parses language instruction based on a pre-trained parser.
        Returns the list of edges of the syntax tree
        Input:
          instr_lang: language instruction [str]
        Output:
          edges_idx: list of edges with node pairs in integer format [list]
          idx2word: maps index to word in instruction [dict]
        """
        instr_parsed_raw = self.predictor.predict(sentence=instr_lang)

        word_seq_dict = {i: w for i, w in enumerate(instr_parsed_raw['words'])}

        instr_padded = instr_parsed_raw['hierplane_tree']['text'] + ' '
        word_indices = self._get_word_indices(instr_padded)

        dependency_tree = instr_parsed_raw['hierplane_tree']['root']

        # Format tree also removes edges_word
        _, edges_idx, depth = self._format_tree(dependency_tree, word_indices)

        return edges_idx, word_seq_dict, depth