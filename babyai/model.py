import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable
from torch_geometric.nn import GCNConv, AGNNConv, GATConv
from torch_geometric.data import Batch, Data
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import BertModel, BertTokenizer
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import AutoModel, AutoTokenizer


import babyai.rl
from babyai.rl.utils.supervised_losses import required_heads


# From https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ImageBOWEmbedding(nn.Module):
   def __init__(self, max_value, embedding_dim):
       super().__init__()
       self.max_value = max_value
       self.embedding_dim = embedding_dim
       self.embedding = nn.Embedding(3 * max_value, embedding_dim)
       self.apply(initialize_parameters)

   def forward(self, inputs):
       offsets = torch.Tensor([0, self.max_value, 2 * self.max_value]).to(inputs.device)
       inputs = (inputs + offsets[None, :, None, None]).long()
       return self.embedding(inputs).sum(1).permute(0, 3, 1, 2)


class GCN(torch.nn.Module):
    def __init__(self, embedding_dim: int, two_layers: bool = False, dropout: bool = False):
        super(GCN, self).__init__()
        self.two_layers = two_layers

        self.dropout = dropout
        self.conv1 = GCNConv(embedding_dim, embedding_dim, normalize=True, add_self_loops=False)
        if two_layers:
            self.conv2 = GCNConv(embedding_dim, embedding_dim, add_self_loops=False)


    def forward(self, data, num_message_rounds=1):
        x, edge_index = data.x, data.edge_index

        for num in range(num_message_rounds):
            x = self.conv1(x, edge_index)
            if self.two_layers:
                x = F.relu(x)
                if self.dropout:
                    x = F.dropout(x, training=self.training)
                x = self.conv2(x, edge_index)
        return x


class AGCN(torch.nn.Module):
    def __init__(self, embedding_dim: int, requires_grad: bool = True):
        super(AGCN, self).__init__()
        self.att1 = AGNNConv(requires_grad=requires_grad)

    def forward(self, data, num_message_rounds=1):
        x, edge_index = data.x, data.edge_index
        for num in range(num_message_rounds):
            x = self.att1(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, embedding_dim: int, two_layers: bool = False, heads: int = 1, dropout: bool = False):
        super(GAT, self).__init__()
        self.two_layers = two_layers
        self.heads = heads
        self.dropout = dropout

        self.att1 = GATConv(in_channels=embedding_dim, out_channels=embedding_dim, heads=heads, add_self_loops=False)

        if two_layers:
            self.att2 = GATConv(in_channels=embedding_dim * heads, out_channels=embedding_dim, heads=1,
                                add_self_loops=False)

    def forward(self, data, num_message_rounds=1):
        x, edge_index = data.x, data.edge_index
        for num in range(num_message_rounds):
            if self.two_layers:
                if self.dropout:
                    x = F.dropout(x, p=0.2, training=self.training)
                x = self.att1(x, edge_index)
                x = F.elu(x)
                if self.dropout:
                    x = F.dropout(x, p=0.2, training=self.training)
                x = self.att2(x, edge_index)
            else:
                x = self.att1(x, edge_index)
        return x


# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class FiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=imm_channels,
            kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(
            in_channels=imm_channels, out_channels=out_features,
            kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        weight = self.weight(y).unsqueeze(2).unsqueeze(3)
        bias = self.bias(y).unsqueeze(2).unsqueeze(3)
        out = x * weight + bias
        return F.relu(self.bn2(out))


class DoubleFILM(nn.Module):
    # from https://arxiv.org/pdf/1806.01946.pdf
    def __init__(self, in_instr_size, in_image_size, out_image_instr_size, batch_norm=True):
        super().__init__()
        self.instr_size = in_instr_size
        self.in_image_size = in_image_size
        self.out_image_instr_size = out_image_instr_size
        self.batch_norm = batch_norm

        self.trans_text = nn.Linear(in_instr_size, out_image_instr_size)
        self.conv = nn.Conv2d(in_channels=self.in_image_size,
                              out_channels=self.out_image_instr_size,
                              kernel_size=(3, 3), padding=1)

        # Input = instruction embedding size, output = 2 * image output size
        self.gamma_beta_trans = nn.Linear(in_instr_size, 2 * out_image_instr_size)
        self.gamma_beta_conv = nn.Conv2d(in_image_size, 2 * out_image_instr_size,
                                         kernel_size=(3, 3), padding=1)

        if self.batch_norm:
            self.bn_image = nn.BatchNorm2d(out_image_instr_size)
            self.bn_mod_image = nn.BatchNorm2d(out_image_instr_size)
            self.bn_mod_instr = nn.BatchNorm2d(out_image_instr_size)

        self.apply(initialize_parameters)

    def forward(self, image, instr):
        # 1: pass image input through a CNN
        if self.batch_norm:
            image_conv = self.bn_image(self.conv(image))
        else:
            image_conv = self.conv(image)

        # 2: pass instr through a linear layer
        text_trans = self.trans_text(instr).unsqueeze(2).unsqueeze(2).expand_as(image_conv)

        # 3: generate gamma, beta for image via CNN 2
        gamma_conv, beta_conv = torch.chunk(self.gamma_beta_conv(image), 2, dim=1)
        # 4: generate gamma, beta for instr via linear layer 2
        gamma_trans, beta_trans = torch.chunk(self.gamma_beta_trans(instr), 2, dim=1)

        gamma_trans = gamma_trans.unsqueeze(2).unsqueeze(2).expand_as(image_conv)
        beta_trans = beta_trans.unsqueeze(2).unsqueeze(2).expand_as(image_conv)

        # 5: modulate image with instr gamma, beta and instr with image gamma, beta
        if self.batch_norm:
            image_modulated_with_text = F.relu(self.bn_mod_image((1 + gamma_trans) * image_conv + beta_trans))
            text_modulated_with_image = F.relu(self.bn_mod_instr((1 + gamma_conv) * text_trans + beta_conv))
        else:
            image_modulated_with_text = F.relu((1 + gamma_trans) * image_conv + beta_trans)
            text_modulated_with_image = F.relu((1 + gamma_conv) * text_trans + beta_conv)

        # 6: concatenate modulated representations
        mix = image_modulated_with_text + text_modulated_with_image
        return mix


class ACModel(nn.Module, babyai.rl.RecurrentACModel):
    def __init__(self, obs_space, action_space,
                 image_dim=128, memory_dim=128, instr_dim=128,
                 use_instr=False, lang_model="gru", use_memory=False, arch="film_endpool_res", aux_info=None,
                 gie_pretrained_emb="random", gie_freeze_emb=False, gie_aggr_method="mean",
                 gie_message_rounds=-1, gie_two_layers=False, gie_heads=1, device="cuda"):
        super().__init__()

        endpool = 'endpool' in arch
        use_bow = 'bow' in arch
        pixel = 'pixel' in arch
        self.res = 'res' in arch

        # Decide which components are enabled
        self.use_instr = use_instr
        self.use_memory = use_memory
        self.arch = arch
        self.lang_model = lang_model
        self.aux_info = aux_info
        if self.res and image_dim != 128:
            raise ValueError(f"image_dim is {image_dim}, expected 128")
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim

        self.obs_space = obs_space

        self.device = device
        self.gie_pretrained_emb = gie_pretrained_emb
        self.gie_freeze_emb = gie_freeze_emb
        self.gie_aggr_method = gie_aggr_method

        if gie_message_rounds < 0 and self.gie_aggr_method == 'mean':
            self.gie_message_rounds = 1
        elif gie_message_rounds < 0 and self.gie_aggr_method == 'root':
            self.gie_message_rounds = 0
        else:
            self.gie_message_rounds = gie_message_rounds

        self.gie_two_layers = gie_two_layers
        self.gie_heads = gie_heads

        for part in self.arch.split('_'):
            if part not in ['bow', 'pixels', 'endpool', 'res', 'film', 'film2', 'cnn']:
                raise ValueError("Incorrect architecture name: {}".format(self.arch))

        # BabyAI 1.0 env. state encoder
        if 'cnn' in self.arch:
            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=image_dim, kernel_size=(2, 2)),
                nn.ReLU()
            )
        # This image_conv architecture is the same irrespective of FILM or FILM**2
        elif 'film' in self.arch:
            if not self.use_instr:
                raise ValueError("FiLM architecture can be used when instructions are enabled")

            self.image_conv = nn.Sequential(*[
                *([ImageBOWEmbedding(obs_space['image'], 128)] if use_bow else []),
                *([nn.Conv2d(
                    in_channels=3, out_channels=128, kernel_size=(8, 8),
                    stride=8, padding=0)] if pixel else []),
                nn.Conv2d(
                    in_channels=128 if use_bow or pixel else 3, out_channels=128,
                    kernel_size=(3, 3) if endpool else (2, 2), stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)])
            ])
            self.film_pool = nn.MaxPool2d(kernel_size=(7, 7) if endpool else (2, 2), stride=2)
        else:
            raise ValueError("Incorrect architecture name: {}".format(arch))

        # Define instruction embedding
        if self.use_instr:
            if 'gie' in self.lang_model:
                if 'bert' in self.gie_pretrained_emb:
                    embedding_dim = -1
                    self.tokens_cache = {}
                    self.tokens_cache_hits = 0
                    self.tokens_cache_misses = 0

                    if self.gie_pretrained_emb == 'bert':
                        self.word_embedding_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                                                      do_lower_case=True)
                        self.word_embedding = BertModel.from_pretrained('bert-base-uncased')
                        embedding_dim = 768

                    elif self.gie_pretrained_emb == 'distil_bert':
                        self.word_embedding_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',
                                                                                           do_lower_case=True)
                        self.word_embedding = DistilBertModel.from_pretrained('distilbert-base-uncased')
                        embedding_dim = 768

                    elif self.gie_pretrained_emb == 'tiny_bert':
                        self.word_embedding_tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
                        self.word_embedding = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
                        embedding_dim = 128

                    # reduce BERT dimensionality
                    if embedding_dim > 128:
                        self.embedding_reducer = nn.Sequential(
                            nn.Linear(embedding_dim, self.instr_dim),
                            nn.Tanh()
                        )
                    else:
                        self.embedding_reducer = None

                else:  # randomly initialise embeddings
                    self.word_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)
                    torch.nn.init.xavier_uniform_(self.word_embedding.weight)

                self.add_module('Word_Embedding', self.word_embedding)

                # initialise graph network
                if 'gie_gat' in self.lang_model:
                    self.gnn = GAT(self.instr_dim, two_layers=self.gie_two_layers, heads=self.gie_heads, dropout=False)
                    self.add_module('GIE_GAT', self.gnn)
                else:
                    self.gnn = GCN(self.instr_dim, two_layers=self.gie_two_layers, dropout=False)
                    self.add_module('GCN', self.gnn)
                self.final_instr_dim = self.instr_dim

            elif self.lang_model in ['gru', 'gru_bert', 'bigru', 'attgru', 'attgru_bert']:

                if 'bert' in self.lang_model:
                    self.word_embedding_tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
                    self.word_embedding = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
                else:
                    self.word_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)

                gru_dim = self.instr_dim
                if self.lang_model in ['bigru', 'attgru', 'attgru_bert']:
                    gru_dim //= 2
                self.instr_rnn = nn.GRU(
                    self.instr_dim, gru_dim, batch_first=True,
                    bidirectional=(self.lang_model in ['bigru', 'attgru', 'attgru_bert']))
                self.final_instr_dim = self.instr_dim

            if "att" in self.lang_model:
                self.memory2key = nn.Linear(self.memory_size, self.final_instr_dim)

            if 'film' in self.arch:
                num_module = 2
                self.controllers = []
                for ni in range(num_module):
                    if 'film2' in self.arch:
                        mod = DoubleFILM(in_instr_size=self.final_instr_dim, in_image_size=128,
                                         out_image_instr_size=128 if ni < num_module - 1 else self.image_dim)
                    else:
                        mod = FiLM(
                            in_features=self.final_instr_dim,
                            out_features=128 if ni < num_module-1 else self.image_dim,
                            in_channels=128, imm_channels=128)
                    self.controllers.append(mod)
                    self.add_module('FiLM_' + str(ni), mod)

        # Define memory and resize image embedding
        self.embedding_size = self.image_dim
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_dim, self.memory_dim)
            self.embedding_size = self.semi_memory_size

        if self.use_instr and "film" not in arch:
            self.embedding_size += self.final_instr_dim

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

        # Define head for extra info
        if self.aux_info:
            self.extra_heads = None
            self.add_heads()

    def add_heads(self):
        '''
        When using auxiliary tasks, the environment yields at each step some binary, continous, or multiclass
        information. The agent needs to predict those information. This function add extra heads to the model
        that output the predictions. There is a head per extra information (the head type depends on the extra
        information type).
        '''
        self.extra_heads = nn.ModuleDict()
        for info in self.aux_info:
            if required_heads[info] == 'binary':
                self.extra_heads[info] = nn.Linear(self.embedding_size, 1)
            elif required_heads[info].startswith('multiclass'):
                n_classes = int(required_heads[info].split('multiclass')[-1])
                self.extra_heads[info] = nn.Linear(self.embedding_size, n_classes)
            elif required_heads[info].startswith('continuous'):
                if required_heads[info].endswith('01'):
                    self.extra_heads[info] = nn.Sequential(nn.Linear(self.embedding_size, 1), nn.Sigmoid())
                else:
                    raise ValueError('Only continous01 is implemented')
            else:
                raise ValueError('Type not supported')
            # initializing these parameters independently is done in order to have consistency of results when using
            # supervised-loss-coef = 0 and when not using any extra binary information
            self.extra_heads[info].apply(initialize_parameters)

    def add_extra_heads_if_necessary(self, aux_info):
        '''
        This function allows using a pre-trained model without aux_info and add aux_info to it and still make
        it possible to finetune.
        '''
        try:
            if not hasattr(self, 'aux_info') or not set(self.aux_info) == set(aux_info):
                self.aux_info = aux_info
                self.add_heads()
        except Exception:
            raise ValueError('Could not add extra heads')

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forward(self, obs, memory, instr_embedding=None):
        if self.use_instr and instr_embedding is None:
            # obs.instr will be a tuple of (edge_indices, words_indices) if using gie
            instr_embedding = self._get_instr_embedding(obs.instr)

        if self.use_instr and "att" in self.lang_model:
            # outputs: B x L x D
            # memory: B x M

            if "gie" in self.lang_model:
                mask = torch.ones_like(instr_embedding[:, :, 0])
            elif "bert" in self.lang_model:
                mask = (obs.instr[1] != 0).float()
                mask = mask[:, :instr_embedding.shape[1]]
                instr_embedding = instr_embedding[:, :mask.shape[1]]
            else:
                mask = (obs.instr != 0).float()
                # The mask tensor has the same length as obs.instr, and
                # thus can be both shorter and longer than instr_embedding.
                # It can be longer if instr_embedding is computed
                # for a subbatch of obs.instr.
                # It can be shorter if obs.instr is a subbatch of
                # the batch that instr_embeddings was computed for.
                # Here, we make sure that mask and instr_embeddings
                # have equal length along dimension 1.
                mask = mask[:, :instr_embedding.shape[1]]
                instr_embedding = instr_embedding[:, :mask.shape[1]]

            keys = self.memory2key(memory)
            pre_softmax = (keys[:, None, :] * instr_embedding).sum(2) + 1000 * mask
            attention = F.softmax(pre_softmax, dim=1)
            instr_embedding = (instr_embedding * attention[:, :, None]).sum(1)

        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        if 'pixel' in self.arch:
            x /= 256.0

        x = self.image_conv(x)

        if 'film' in self.arch:
            for controller in self.controllers:
                out = controller(x, instr_embedding)
                if self.res:
                    out += x
                x = out

            if 'film2' in self.arch:
                x = x.max(3)[0].max(2)[0]
            else:
                x = self.film_pool(x)
            x = F.relu(x)

        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_instr and "film" not in self.arch:
            embedding = torch.cat((embedding, instr_embedding), dim=1)

        if hasattr(self, 'aux_info') and self.aux_info:
            extra_predictions = {info: self.extra_heads[info](embedding) for info in self.extra_heads}
        else:
            extra_predictions = dict()

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return {'dist': dist, 'value': value, 'memory': memory, 'extra_predictions': extra_predictions}

    def _get_instr_embedding(self, instr):
        if 'gie' in self.lang_model:
            edges_indices_t, words_indices_t, depths_t, missions_tokens = instr

            if 'bert' in self.gie_pretrained_emb:
                tokens_ids_t = self.word_embedding_tokenizer.batch_encode_plus(
                    missions_tokens.tolist(), is_pretokenized=True, return_tensors='pt').to(self.device)

                # Note that we remove the first and last embedding per sequence, as they represent START and END
                # tokens to BERT, but would then mess up our edge indices for the graph
                node_embeddings_unreduced = self.word_embedding(**tokens_ids_t)[0][:, 1:-1, :]

                # Note that freezing the embeddings ONLY freezes the underlying embedding store, NOT the reducer
                if self.gie_freeze_emb:
                    node_embeddings_unreduced.detach()

                if self.embedding_reducer:
                    node_embeddings = self.embedding_reducer(node_embeddings_unreduced)
                else:
                    node_embeddings = node_embeddings_unreduced
            else:
                node_embeddings = self.word_embedding(words_indices_t)


                if self.gie_freeze_emb:
                    node_embeddings.detach()

            # create a batch for GCN
            data = []
            for i, nodes in enumerate(node_embeddings):
                # nodes of shape [num, nodes, num_node_features]
                data.append(Data(x=nodes, edge_index=edges_indices_t[i].t().contiguous(), num_nodes=len(nodes)))

            batch = Batch.from_data_list(data)

            if self.gie_message_rounds == 0:
                gie_embeddings = self.gnn(batch, num_message_rounds=depths_t.max().item())
            else:
                gie_embeddings = self.gnn(batch, num_message_rounds=self.gie_message_rounds)

            # reshape from [batch_size * n_nodes, instr_dim] -> [batch_size, n_nodes, instr_dim]
            gie_embeddings_reshaped = gie_embeddings.view(batch.num_graphs, -1, batch.num_features)

            # return average of node embeddings if 'mean' otherwise root embedding
            if "att" in self.lang_model:
                return gie_embeddings_reshaped
            else:
                if self.gie_aggr_method == "mean":
                    return torch.mean(gie_embeddings_reshaped, dim=1)
                elif self.gie_aggr_method == "root":
                    return gie_embeddings_reshaped[:, 0, :]
                elif self.gie_aggr_method == "max":
                    idxs = torch.norm(gie_embeddings_reshaped, dim=2).argmax(dim=1)
                    return torch.gather(gie_embeddings_reshaped, 1, idxs.unsqueeze(-1).repeat(1, self.instr_dim).unsqueeze(1))[:, 0, :]

                else:
                    raise ValueError(f'wrong gie aggregation method: {self.gie_aggr_method}')

        elif self.lang_model in ['gru', 'gru_bert']:
            if self.lang_model == 'gru_bert':

                edges_indices_t, words_indices_t, depths_t, missions_tokens = instr
                lengths = (words_indices_t != 0).sum(1).long()

                tokens_ids_t = self.word_embedding_tokenizer.batch_encode_plus(
                    missions_tokens.tolist(), is_pretokenized=True, return_tensors='pt').to(self.device)

                node_embeddings_unreduced = self.word_embedding(**tokens_ids_t)[0][:, 1:-1, :]
                node_embeddings_unreduced.detach()
                x = node_embeddings_unreduced
            else:
                lengths = (instr != 0).sum(1).long()
                x = self.word_embedding(instr)
            out, _ = self.instr_rnn(x)
            hidden = out[range(len(lengths)), lengths-1, :]
            return hidden

        elif self.lang_model in ['bigru', 'attgru', 'attgru_bert']:

            if 'bert' in self.lang_model:
                edges_indices_t, words_indices_t, depths_t, missions_tokens = instr

                instr = words_indices_t
                tokens_ids_t = self.word_embedding_tokenizer.batch_encode_plus(
                    missions_tokens.tolist(), is_pretokenized=True, return_tensors='pt').to(self.device)

                node_embeddings_unreduced = self.word_embedding(**tokens_ids_t)[0][:, 1:-1, :]
                node_embeddings_unreduced.detach()
                x = node_embeddings_unreduced

            lengths = (instr != 0).sum(1).long()
            masks = (instr != 0).float()

            if lengths.shape[0] > 1:
                seq_lengths, perm_idx = lengths.sort(0, descending=True)
                iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0)
                if instr.is_cuda:
                    iperm_idx = iperm_idx.cuda()
                for i, v in enumerate(perm_idx):
                    iperm_idx[v.data] = i

                if 'bert' not in self.lang_model:
                    x = self.word_embedding(instr)

                inputs = x # self.word_embedding(instr)
                inputs = inputs[perm_idx]

                inputs = pack_padded_sequence(inputs, seq_lengths.data.cpu().numpy(), batch_first=True)

                outputs, final_states = self.instr_rnn(inputs)
            else:
                instr = instr[:, 0:lengths[0]]

                if 'bert' not in self.lang_model:
                    x = self.word_embedding(instr)

                outputs, final_states = self.instr_rnn(x)
                iperm_idx = None

            final_states = final_states.transpose(0, 1).contiguous()
            final_states = final_states.view(final_states.shape[0], -1)
            if iperm_idx is not None:
                outputs, _ = pad_packed_sequence(outputs, batch_first=True)
                outputs = outputs[iperm_idx]
                final_states = final_states[iperm_idx]

            return outputs if 'attgru' in self.lang_model else final_states

        else:
            ValueError("Undefined instruction architecture: {}".format(self.use_instr))
