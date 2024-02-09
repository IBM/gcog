import torch
import gcog.task.config as config
import math
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

class PositionalEncoding(torch.nn.Module):
    """
    Positional encoding
    """
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[0,:x.size(1)]
        return self.dropout(x)

# From here:https://jaketae.github.io/study/relative-positional-encoding/
class RelativeGlobalAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, max_len=1024, dropout=0.0):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError(
                "incompatible `d_model` and `num_heads`"
            )
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.query = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.Er = torch.nn.Parameter(torch.randn(max_len, d_head))
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(max_len, max_len))
            .unsqueeze(0).unsqueeze(0)
        )

    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        if seq_len > self.max_len:
            raise ValueError(
                "sequence length exceeds model capacity"
            )
        
        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape = (batch_size, num_heads, seq_len, d_head)
        
        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        
        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        return self.dropout(out)
        
    
    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel


class TransformerBlock(torch.nn.Module):
    """
    Transformer block
    """
    def __init__(self,
                 dim_input,
                 n_tokens,
                 positional_encoding='absolute',
                 nhead=1,
                 embedding_dim=256,
                 dropout=0,
                 learning_rate=0.0001,
                 lossfunc='CrossEntropy'):
        super(TransformerBlock,self).__init__()

        # Define general parameters
        self.dim_input = dim_input 
        self.nhead = nhead
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.positional_encoding = positional_encoding
        
        # a linear embedding
        self.w_embed = torch.nn.Linear(self.dim_input,self.embedding_dim)
        self.dropout_embed = torch.nn.Dropout(p=dropout)

        # positional encoding
        if positional_encoding=='absolute':
            self.pe = PositionalEncoding(self.embedding_dim, max_len=n_tokens)
            self.selfattention = torch.nn.MultiheadAttention(self.embedding_dim,nhead,dropout,batch_first=True)
        elif positional_encoding=='relative':
            self.selfattention = RelativeGlobalAttention(self.embedding_dim, nhead, dropout=dropout, max_len=n_tokens)

        #self.mlp = torch.nn.Sequential(
        #    torch.nn.Linear(self.embedding_dim,4*self.embedding_dim),
        #    torch.nn.ReLU(),
        #    torch.nn.Linear(4*self.embedding_dim,self.embedding_dim),
        #    torch.nn.ReLU()
        #)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim,self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim,self.embedding_dim),
            torch.nn.ReLU()
        )

        # layer norm; 1st is after attention (embedding dim); 2nd is after RNN 
        self.layernorm0 = torch.nn.LayerNorm(self.embedding_dim)
        self.layernorm1 = torch.nn.LayerNorm(self.embedding_dim)

    def forward(self, input, noise=False, dropout=False):
        """
        Run a forward pass of a trial by input_elements matrix
        For each time window, pass each 
        input (Tensor): batch x seq_length x dim_input x time
        """
        device = input.device
        #Add noise to inputs
        if noise:
            input = input + torch.randn(input.shape, device=device, dtype=torch.float)/5

        ####
        # transformer block
        embedding = self.w_embed(input)
        if self.positional_encoding in ['relative']:
            attn_outputs = self.selfattention(embedding)
        else:
            embedding = self.pe(embedding) # positional encoding
            embedding = self.dropout_embed(embedding)
            attn_outputs, attn_out_weights = self.selfattention(embedding, embedding, embedding, need_weights=False)
        attn_outputs = self.layernorm0(attn_outputs+embedding) 
        transformer_out = self.mlp(attn_outputs)
        transformer_out = self.layernorm1(transformer_out+attn_outputs)

        return transformer_out

class Transformer(torch.nn.Module):
    def __init__(self,
                 num_rule_inputs=config.RULE_ARRAY_LENGTH,
                 num_stim_inputs=config.STIMULI_ARRAY_LENGTH, # (26 shapes + 10 colors)
                 nhead=1,
                 max_tree_depth=100,
                 embedding_dim=256,
                 num_hidden=256,
                 num_rnn_layers=1,
                 num_motor_decision_outputs=138,
                 positional_encoding='absolute',
                 dropout=0,
                 learning_rate=0.0001,
                 lossfunc='CrossEntropy'):
        super(Transformer,self).__init__()

        # Define general parameters
        self.num_rule_inputs = num_rule_inputs
        self.num_stim_inputs =  num_stim_inputs
        self.num_inputs = num_rule_inputs + num_stim_inputs
        # typically 4x the size of the embedding dimension. 
        # this is the dim expansion of the hidden layer of the FNN, after attention
        #self.num_hidden = self.num_inputs*4 if num_hidden is None else num_hidden
        
        #self.embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.num_hidden = num_hidden
        self.num_rnn_layers = num_rnn_layers
        self.num_motor_decision_outputs = num_motor_decision_outputs
        
        self.nhead = nhead
        self.dropout = dropout
        
        #### Stim transformer block
        max_tokens_stim = config.GRIDSIZE_X * config.GRIDSIZE_Y + 1 # INCLUDE EOS
        self.stim_transformer = TransformerBlock(self.num_stim_inputs,
                                                 max_tokens_stim,
                                                 nhead=self.nhead,
                                                 embedding_dim=self.embedding_dim,
                                                 dropout=self.dropout,
                                                 learning_rate=learning_rate,
                                                 lossfunc=lossfunc)

        #### Rule transformer block
        max_tokens_rule = max_tree_depth
        self.rule_transformer = TransformerBlock(self.num_rule_inputs,
                                                 max_tokens_rule,
                                                 nhead=self.nhead,
                                                 embedding_dim=int(embedding_dim),
                                                 dropout=self.dropout,
                                                 positional_encoding=positional_encoding,
                                                 learning_rate=learning_rate,
                                                 lossfunc=lossfunc)

        # rule_fnn
        self.rule_fnn = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim,self.num_hidden),
            torch.nn.ReLU()
        )

        # stim fnn
        self.stim_fnn = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim,self.num_hidden),
            torch.nn.ReLU()
        )

        self.fnn_layernorm = torch.nn.LayerNorm(self.num_hidden)
        self.fnn_dropout = torch.nn.Dropout(p=dropout)

        # multimodal fnn
        self.fnn = torch.nn.Sequential(
            torch.nn.Linear(self.num_hidden,self.num_hidden*2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.num_hidden*2,self.num_hidden*2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.num_hidden*2,self.num_hidden),
            torch.nn.ReLU()
        )
        self.fnnout_layernorm = torch.nn.LayerNorm(self.num_hidden)
        self.fnnout_dropout = torch.nn.Dropout(p=dropout)

        self.w_out = torch.nn.Sequential(
            torch.nn.Linear(self.num_hidden,self.num_motor_decision_outputs),
            torch.nn.LogSoftmax(dim=-1)
        )
        
        # Define loss function
        if lossfunc=='MSE':
            self.lossfunc = torch.nn.MSELoss(reduction='none')
        if lossfunc=='CrossEntropy':
            self.lossfunc = torch.nn.CrossEntropyLoss()
        if lossfunc=='NLL':
            self.lossfunc = torch.nn.NLLLoss()

        # Construct optimizer
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate,weight_decay=0.01)
        
    def forward(self, rule_inputs, stim_inputs, noise=False, dropout=False):
        assert(rule_inputs.device==stim_inputs.device)
        device = rule_inputs.device
        #Add noise to inputs
        if noise:
            rule_inputs = rule_inputs + torch.randn(rule_inputs.shape, device=device, dtype=torch.float)/5
            stim_inputs = stim_inputs + torch.randn(stim_inputs.shape, device=device, dtype=torch.float)/5

        ####
        # Rule input first
        # initialize RNN state 
        seq_length = rule_inputs.size()[1]
        batch_size = rule_inputs.size()[0]
        hn = torch.zeros(self.num_rnn_layers,
                         batch_size,
                         self.num_hidden, 
                         device=device)
        # rule transformer block
        rule_transformer_out = self.rule_transformer(rule_inputs,noise=noise,dropout=dropout)

        assert(stim_inputs.shape[3]==1) # no multiple screens
        stim_transformer_out = self.stim_transformer(stim_inputs[:,:,:,0],noise=noise,dropout=dropout)

        rule_fnn_out = self.rule_fnn(rule_transformer_out)[:,-1,:] # get end of sequence
        stim_fnn_out = self.stim_fnn(stim_transformer_out)[:,-1,:] # get end of sequence

        fnn_out = self.fnn_layernorm(rule_fnn_out + stim_fnn_out)
        fnn_out = self.fnn_dropout(fnn_out)

        fnn_out = self.fnn(fnn_out)
        fnn_out = self.fnnout_layernorm(fnn_out)
        fnn_out = self.fnnout_dropout(fnn_out)

        outputs = self.w_out(fnn_out)
        return outputs, fnn_out # only want the last output from seq data

class MultimodalTransformer(torch.nn.Module):
    def __init__(self,
                 num_rule_inputs=config.RULE_ARRAY_LENGTH,
                 num_stim_inputs=config.STIMULI_ARRAY_LENGTH, # (26 shapes + 10 colors)
                 nhead=1,
                 nblocks=1,
                 max_tree_depth=100,
                 embedding_dim=256,
                 num_hidden=256,
                 num_motor_decision_outputs=138,
                 dropout=0,
                 positional_encoding='absolute',
                 learning_rate=0.0001,
                 lossfunc='CrossEntropy'):
        super(MultimodalTransformer,self).__init__()

        # Define general parameters
        self.num_rule_inputs = num_rule_inputs
        self.num_stim_inputs =  num_stim_inputs
        self.num_inputs = num_rule_inputs + num_stim_inputs
        
        #self.embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.num_hidden = num_hidden
        self.num_motor_decision_outputs = num_motor_decision_outputs
        
        self.nhead = nhead
        self.dropout = dropout
        
        #### multimodal transformer block
        max_tokens_stim = config.GRIDSIZE_X * config.GRIDSIZE_Y + 1 # INCLUDE EOS
        max_tokens = max_tree_depth + max_tokens_stim
        self.blocks = torch.nn.Sequential(*[TransformerBlock(self.num_inputs,
                                                             max_tokens,
                                                             nhead=self.nhead,
                                                             embedding_dim=self.embedding_dim,
                                                             dropout=self.dropout,
                                                             positional_encoding=positional_encoding,
                                                             learning_rate=learning_rate,
                                                             lossfunc=lossfunc)
                                            for _ in range(nblocks)])


        # multimodal fnn
        self.fnn = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim,self.num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.num_hidden,self.num_hidden*2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.num_hidden*2,self.num_hidden),
            torch.nn.ReLU())
        self.fnnout_layernorm = torch.nn.LayerNorm(self.num_hidden)
        self.fnnout_dropout = torch.nn.Dropout(p=dropout)

        self.w_out = torch.nn.Sequential(
            torch.nn.Linear(self.num_hidden,self.num_motor_decision_outputs),
            torch.nn.LogSoftmax(dim=-1)
        )
        
        # Define loss function
        if lossfunc=='MSE':
            self.lossfunc = torch.nn.MSELoss(reduction='none')
        if lossfunc=='CrossEntropy':
            self.lossfunc = torch.nn.CrossEntropyLoss()
        if lossfunc=='NLL':
            self.lossfunc = torch.nn.NLLLoss()

        # Construct optimizer
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate,weight_decay=0.01)
        
    def forward(self, rule_inputs, stim_inputs, noise=False, dropout=False):
        assert(rule_inputs.device==stim_inputs.device)
        device = rule_inputs.device
        #Add noise to inputs
        if noise:
            rule_inputs = rule_inputs + torch.randn(rule_inputs.shape, device=device, dtype=torch.float)/5
            stim_inputs = stim_inputs + torch.randn(stim_inputs.shape, device=device, dtype=torch.float)/5

        assert(stim_inputs.shape[3]==1) # no multiple screens
        # multimodal transformer
        stim_token_pad = stim_inputs.shape[1]
        stim_embedding_pad = stim_inputs.shape[2]
        rule_token_pad = rule_inputs.shape[1]
        rule_embedding_pad = rule_inputs.shape[2]
        # right padding last dim (embed_pad) and 2nd last dim (token_pad)
        train_rules_pad = torch.nn.functional.pad(rule_inputs,(0,stim_embedding_pad,0,stim_token_pad),'constant')
        train_stim_pad = torch.nn.functional.pad(stim_inputs[:,:,:,0],(rule_embedding_pad,0,rule_token_pad,0),'constant')
        # merge zeropadded inputs
        combined_inputs = train_rules_pad + train_stim_pad 

        ####
        transformer_out = checkpoint_sequential(self.blocks, segments = len(self.blocks), input = combined_inputs)

        fnn_out = self.fnn(transformer_out)[:,-1,:] # get end of sequence

        fnn_out = self.fnnout_layernorm(fnn_out)
        fnn_out = self.fnnout_dropout(fnn_out)

        outputs = self.w_out(fnn_out)
        return outputs, fnn_out # only want the last output from seq data


class TransformerCrossAttn(torch.nn.Module):
    def __init__(self,
                 num_rule_inputs=config.RULE_ARRAY_LENGTH,
                 num_stim_inputs=config.STIMULI_ARRAY_LENGTH, # (26 shapes + 10 colors)
                 nhead=1,
                 max_tree_depth=100,
                 embedding_dim=256,
                 num_hidden=256,
                 num_motor_decision_outputs=138,
                 positional_encoding='absolute',
                 dropout=0,
                 learning_rate=0.0001,
                 lossfunc='CrossEntropy'):
        super(TransformerCrossAttn,self).__init__()

        # Define general parameters
        self.num_rule_inputs = num_rule_inputs
        self.num_stim_inputs =  num_stim_inputs
        self.num_inputs = num_rule_inputs + num_stim_inputs
        # typically 4x the size of the embedding dimension. 
        # this is the dim expansion of the hidden layer of the FNN, after attention
        #self.num_hidden = self.num_inputs*4 if num_hidden is None else num_hidden
        
        self.embedding_dim = embedding_dim
        self.num_hidden = num_hidden
        self.num_motor_decision_outputs = num_motor_decision_outputs
        
        self.nhead = nhead
        self.dropout = dropout
        
        #### Stim transformer block
        max_tokens_stim = config.GRIDSIZE_X * config.GRIDSIZE_Y + 1 # INCLUDE EOS
        self.stim_transformer = TransformerBlock(self.num_stim_inputs,
                                                 max_tokens_stim,
                                                 nhead=self.nhead,
                                                 embedding_dim=self.embedding_dim,
                                                 dropout=self.dropout,
                                                 learning_rate=learning_rate,
                                                 lossfunc=lossfunc)

        #### Rule transformer block
        max_tokens_rule = max_tree_depth
        self.rule_transformer = TransformerBlock(self.num_rule_inputs,
                                                 max_tokens_rule,
                                                 nhead=self.nhead,
                                                 embedding_dim=int(embedding_dim),
                                                 dropout=self.dropout,
                                                 positional_encoding=positional_encoding,
                                                 learning_rate=learning_rate,
                                                 lossfunc=lossfunc)


        self.rule_crossattention = torch.nn.MultiheadAttention(self.num_hidden,nhead,dropout,
                                                               batch_first=True,
                                                               kdim=self.embedding_dim,
                                                               vdim=self.embedding_dim)
        self.stim_crossattention = torch.nn.MultiheadAttention(self.num_hidden,nhead,dropout,
                                                               batch_first=True,
                                                               kdim=self.embedding_dim,
                                                               vdim=self.embedding_dim)
        self.layernorm_crossattn = torch.nn.LayerNorm(self.num_hidden)

        self.layernorm_latent = torch.nn.LayerNorm(self.num_hidden)

        self.w_out = torch.nn.Sequential(
            torch.nn.Linear(self.num_hidden,self.num_motor_decision_outputs),
            torch.nn.LogSoftmax(dim=-1)
        )
        
        # Define loss function
        if lossfunc=='MSE':
            self.lossfunc = torch.nn.MSELoss(reduction='none')
        if lossfunc=='CrossEntropy':
            self.lossfunc = torch.nn.CrossEntropyLoss()
        if lossfunc=='NLL':
            self.lossfunc = torch.nn.NLLLoss()

        # Construct optimizer
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate,weight_decay=0.01)
        
    def forward(self, rule_inputs, stim_inputs, noise=False, dropout=False):
        assert(rule_inputs.device==stim_inputs.device)
        device = rule_inputs.device
        #Add noise to inputs
        if noise:
            rule_inputs = rule_inputs + torch.randn(rule_inputs.shape, device=device, dtype=torch.float)/5
            stim_inputs = stim_inputs + torch.randn(stim_inputs.shape, device=device, dtype=torch.float)/5

        ####
        # Rule input first
        seq_length = rule_inputs.size()[1]
        batch_size = rule_inputs.size()[0]
        latents = torch.zeros(batch_size,
                             1, # sequence size for output
                             self.num_hidden, 
                             device=device)
        # rule transformer block
        rule_transformer_out = self.rule_transformer(rule_inputs,noise=noise,dropout=dropout)

        # stim transformer block
        assert(stim_inputs.shape[3]==1) # no multiple screens
        stim_transformer_out = self.stim_transformer(stim_inputs[:,:,:,0],noise=noise,dropout=dropout)

        # rule cross attn
        rule_cross_attn, cross_weights = self.rule_crossattention(latents, rule_transformer_out, rule_transformer_out)
        #
        latents = self.layernorm_crossattn(rule_cross_attn+latents)
        # 
        # stim cross attn
        stim_cross_attn, cross_weights = self.stim_crossattention(latents, stim_transformer_out, stim_transformer_out)
        # 
        latents = self.layernorm_crossattn(stim_cross_attn+latents)

        outputs = self.w_out(latents)[:,-1,:]
        return outputs, latents[:,-1,:] # only want the last output from seq data


class Perceiver(torch.nn.Module):
    def __init__(self,
                 num_rule_inputs=config.RULE_ARRAY_LENGTH,
                 num_stim_inputs=config.STIMULI_ARRAY_LENGTH, # (26 shapes + 10 colors)
                 nhead=1,
                 max_tree_depth=100,
                 embedding_dim=256,
                 num_hidden=256,
                 num_motor_decision_outputs=138,
                 positional_encoding='absolute',
                 dropout=0,
                 learning_rate=0.0001,
                 lossfunc='CrossEntropy'):
        super(Perceiver,self).__init__()

        # Define general parameters
        self.num_rule_inputs = num_rule_inputs
        self.num_stim_inputs =  num_stim_inputs
        self.num_inputs = num_rule_inputs + num_stim_inputs
        # typically 4x the size of the embedding dimension. 
        # this is the dim expansion of the hidden layer of the FNN, after attention
        #self.num_hidden = self.num_inputs*4 if num_hidden is None else num_hidden
        
        #self.embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.num_hidden = num_hidden
        self.num_motor_decision_outputs = num_motor_decision_outputs
        
        self.nhead = nhead
        self.dropout = dropout
        
        #### Stim transformer block
        max_tokens_stim = config.GRIDSIZE_X * config.GRIDSIZE_Y + 1 # INCLUDE EOS
        self.stim_transformer = TransformerBlock(self.num_stim_inputs,
                                                 max_tokens_stim,
                                                 nhead=self.nhead,
                                                 embedding_dim=self.embedding_dim,
                                                 dropout=self.dropout,
                                                 learning_rate=learning_rate,
                                                 lossfunc=lossfunc)

        #### Rule transformer block
        max_tokens_rule = max_tree_depth
        self.rule_transformer = TransformerBlock(self.num_rule_inputs,
                                                 max_tokens_rule,
                                                 nhead=self.nhead,
                                                 embedding_dim=int(embedding_dim),
                                                 positional_encoding=positional_encoding,
                                                 dropout=self.dropout,
                                                 learning_rate=learning_rate,
                                                 lossfunc=lossfunc)


        self.rule_crossattention = torch.nn.MultiheadAttention(self.num_hidden,nhead,dropout,
                                                               batch_first=True,
                                                               kdim=self.embedding_dim,
                                                               vdim=self.embedding_dim)
        self.rule_ffn = torch.nn.Sequential(
            torch.nn.Linear(self.num_hidden,self.num_hidden),
            torch.nn.ReLU()
        )
        self.layernorm_rulecrossattn = torch.nn.LayerNorm(self.num_hidden)
        # 
        self.stim_crossattention = torch.nn.MultiheadAttention(self.num_hidden,nhead,dropout,
                                                               batch_first=True,
                                                               kdim=self.embedding_dim,
                                                               vdim=self.embedding_dim)
        self.stim_ffn = torch.nn.Sequential(
            torch.nn.Linear(self.num_hidden,self.num_hidden),
            torch.nn.ReLU()
        )
        self.layernorm_stimcrossattn = torch.nn.LayerNorm(self.num_hidden)

        self.layernorm_latent = torch.nn.LayerNorm(self.num_hidden)
        self.latent_selfattention = torch.nn.MultiheadAttention(self.num_hidden,nhead,dropout,batch_first=True)
        self.latent_ffn = torch.nn.Sequential(
            torch.nn.Linear(self.num_hidden,self.num_hidden),
            torch.nn.ReLU()
        )

        self.w_out = torch.nn.Sequential(
            torch.nn.Linear(self.num_hidden,self.num_motor_decision_outputs),
            torch.nn.LogSoftmax(dim=-1)
        )
        
        # Define loss function
        if lossfunc=='MSE':
            self.lossfunc = torch.nn.MSELoss(reduction='none')
        if lossfunc=='CrossEntropy':
            self.lossfunc = torch.nn.CrossEntropyLoss()
        if lossfunc=='NLL':
            self.lossfunc = torch.nn.NLLLoss()

        # Construct optimizer
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate,weight_decay=0.01)
        
    def forward(self, rule_inputs, stim_inputs, noise=False, dropout=False):
        assert(rule_inputs.device==stim_inputs.device)
        device = rule_inputs.device
        #Add noise to inputs
        if noise:
            rule_inputs = rule_inputs + torch.randn(rule_inputs.shape, device=device, dtype=torch.float)/5
            stim_inputs = stim_inputs + torch.randn(stim_inputs.shape, device=device, dtype=torch.float)/5

        ####
        seq_length = rule_inputs.size()[1]
        batch_size = rule_inputs.size()[0]
        latents = torch.zeros(batch_size,
                             10, # sequence size for output (arbitrary)
                             self.num_hidden, 
                             device=device)
        # rule transformer block
        rule_transformer_out = self.rule_transformer(rule_inputs,noise=noise,dropout=dropout)
        #rule_transformer_out = torch.mean(rule_transformer_out,axis=1)

        # stim transformer block
        assert(stim_inputs.shape[3]==1) # no multiple screens
        stim_transformer_out = self.stim_transformer(stim_inputs[:,:,:,0],noise=noise,dropout=dropout)

        # rule cross attn
        rule_cross_attn, cross_weights = self.rule_crossattention(latents, rule_transformer_out, rule_transformer_out)
        rule_cross_attn = self.layernorm_rulecrossattn(rule_cross_attn)
        rule_cross_attn = self.rule_ffn(rule_cross_attn) + rule_cross_attn
        #
        latents = self.layernorm_latent(rule_cross_attn+latents)
        # self attention latents
        latent_attn, latent_weights = self.latent_selfattention(latents,latents,latents,need_weights=False)
        latents = self.layernorm_latent(latent_attn+latents)
        # 
        # stim cross attn
        stim_cross_attn, cross_weights = self.stim_crossattention(latents, stim_transformer_out, stim_transformer_out)
        stim_cross_attn = self.layernorm_stimcrossattn(stim_cross_attn)
        stim_cross_attn = self.stim_ffn(stim_cross_attn) + stim_cross_attn
        # 
        latents = self.layernorm_latent(stim_cross_attn+latents)
        # self attention latents
        latent_attn, latent_weights = self.latent_selfattention(latents,latents,latents,need_weights=False)
        latents = self.layernorm_latent(latent_attn+latents)
        latents = self.latent_ffn(latents)

        outputs = self.w_out(latents)[:,-1,:]
        return outputs, latents[:,-1,:] # only want the last output from seq data

class SimpleCrossAttn(torch.nn.Module):
    def __init__(self,
                 num_rule_inputs=config.RULE_ARRAY_LENGTH,
                 num_stim_inputs=config.STIMULI_ARRAY_LENGTH, # (26 shapes + 10 colors)
                 nhead=1,
                 max_tree_depth=100,
                 embedding_dim=256,
                 num_hidden=256,
                 num_motor_decision_outputs=138,
                 positional_encoding='absolute',
                 dropout=0,
                 learning_rate=0.0001,
                 lossfunc='CrossEntropy'):
        super(SimpleCrossAttn,self).__init__()

        # Define general parameters
        self.num_rule_inputs = num_rule_inputs
        self.num_stim_inputs =  num_stim_inputs
        self.num_inputs = num_rule_inputs + num_stim_inputs
        # typically 4x the size of the embedding dimension. 
        # this is the dim expansion of the hidden layer of the FNN, after attention
        #self.num_hidden = self.num_inputs*4 if num_hidden is None else num_hidden
        
        self.embedding_dim = embedding_dim
        self.num_hidden = num_hidden
        self.num_motor_decision_outputs = num_motor_decision_outputs
        
        self.nhead = nhead
        self.dropout = dropout
        
        #### Stim transformer block
        max_tokens_stim = config.GRIDSIZE_X * config.GRIDSIZE_Y + 1 # INCLUDE EOS
        self.stim_transformer = TransformerBlock(self.num_stim_inputs,
                                                 max_tokens_stim,
                                                 nhead=self.nhead,
                                                 embedding_dim=self.embedding_dim,
                                                 dropout=self.dropout,
                                                 learning_rate=learning_rate,
                                                 lossfunc=lossfunc)

        #### Rule transformer block
        max_tokens_rule = max_tree_depth
        self.rule_transformer = TransformerBlock(self.num_rule_inputs,
                                                 max_tokens_rule,
                                                 nhead=self.nhead,
                                                 embedding_dim=int(embedding_dim),
                                                 dropout=self.dropout,
                                                 positional_encoding=positional_encoding,
                                                 learning_rate=learning_rate,
                                                 lossfunc=lossfunc)


        self.crossattention = torch.nn.MultiheadAttention(self.embedding_dim,nhead,dropout,
                                                          batch_first=True,
                                                          kdim=self.embedding_dim,
                                                          vdim=self.embedding_dim)
        self.layernorm_crossattn = torch.nn.LayerNorm(self.embedding_dim)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim,self.num_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.num_hidden,self.num_hidden),
            torch.nn.ReLU()
        )
        self.layernorm_mlp = torch.nn.LayerNorm(self.num_hidden)

        self.w_out = torch.nn.Sequential(
            torch.nn.Linear(self.num_hidden,self.num_motor_decision_outputs),
            torch.nn.LogSoftmax(dim=-1)
        )
        
        # Define loss function
        if lossfunc=='MSE':
            self.lossfunc = torch.nn.MSELoss(reduction='none')
        if lossfunc=='CrossEntropy':
            self.lossfunc = torch.nn.CrossEntropyLoss()
        if lossfunc=='NLL':
            self.lossfunc = torch.nn.NLLLoss()

        # Construct optimizer
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate,weight_decay=0.01)
        
    def forward(self, rule_inputs, stim_inputs, noise=False, dropout=False):
        assert(rule_inputs.device==stim_inputs.device)
        device = rule_inputs.device
        #Add noise to inputs
        if noise:
            rule_inputs = rule_inputs + torch.randn(rule_inputs.shape, device=device, dtype=torch.float)/5
            stim_inputs = stim_inputs + torch.randn(stim_inputs.shape, device=device, dtype=torch.float)/5

        ####
        # Rule input first
        seq_length = rule_inputs.size()[1]
        batch_size = rule_inputs.size()[0]

        # rule transformer block
        rule_transformer_out = self.rule_transformer(rule_inputs,noise=noise,dropout=dropout)
        #rule_transformer_out = torch.mean(rule_transformer_out,axis=1)

        # stim transformer block
        assert(stim_inputs.shape[3]==1) # no multiple screens
        stim_transformer_out = self.stim_transformer(stim_inputs[:,:,:,0],noise=noise,dropout=dropout)

        # rule cross attn
        cross_attn, cross_weights = self.crossattention(rule_transformer_out, stim_transformer_out, stim_transformer_out)
        #
        mlp_in = self.layernorm_crossattn(cross_attn)+rule_transformer_out
        # 
        mlp_out = self.mlp(mlp_in)
        mlp_out = self.layernorm_mlp(mlp_out)
        # 
        outputs = self.w_out(mlp_out[:,-1,:])
        return outputs, mlp_in[:,-1,:] # only want the last output from seq data


class SSTFMR(torch.nn.Module):
    def __init__(self,
                 num_rule_inputs=config.RULE_ARRAY_LENGTH,
                 num_stim_inputs=config.STIMULI_ARRAY_LENGTH, # (26 shapes + 10 colors)
                 nhead=1,
                 nblocks=1,
                 max_tree_depth=100,
                 embedding_dim=256,
                 num_hidden=256,
                 num_motor_decision_outputs=138,
                 dropout=0,
                 positional_encoding='absolute',
                 learning_rate=0.0001,
                 lossfunc='CrossEntropy'):
        super(SSTFMR,self).__init__()

        # Define general parameters
        self.num_rule_inputs = num_rule_inputs
        self.num_stim_inputs =  num_stim_inputs
        self.num_inputs = num_rule_inputs + num_stim_inputs
        
        self.embedding_dim = embedding_dim
        self.num_hidden = num_hidden
        self.num_motor_decision_outputs = num_motor_decision_outputs
        
        self.nhead = nhead
        self.dropout = dropout
        
        # a linear embedding
        self.w_embed = torch.nn.Linear(self.num_inputs,self.embedding_dim)

        #### multimodal transformer block
        max_tokens_stim = config.GRIDSIZE_X * config.GRIDSIZE_Y + 1 # INCLUDE EOS
        max_tokens = max_tree_depth + max_tokens_stim
        self.blocks = torch.nn.Sequential(*[TransformerBlock2(embedding_dim,
                                                              max_tokens,
                                                              nhead=self.nhead,
                                                              dropout=self.dropout,
                                                              positional_encoding=positional_encoding,
                                                              learning_rate=learning_rate,
                                                              lossfunc=lossfunc)
                                            for _ in range(nblocks)])


        self.w_out = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim,self.num_motor_decision_outputs),
            torch.nn.LogSoftmax(dim=-1)
        )
        
        # Define loss function
        if lossfunc=='MSE':
            self.lossfunc = torch.nn.MSELoss(reduction='none')
        if lossfunc=='CrossEntropy':
            self.lossfunc = torch.nn.CrossEntropyLoss()
        if lossfunc=='NLL':
            self.lossfunc = torch.nn.NLLLoss()

        # Construct optimizer
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate,weight_decay=0.01)
        
    def forward(self, rule_inputs, stim_inputs, noise=False, dropout=False):
        assert(rule_inputs.device==stim_inputs.device)
        device = rule_inputs.device
        #Add noise to inputs
        if noise:
            rule_inputs = rule_inputs + torch.randn(rule_inputs.shape, device=device, dtype=torch.float)/5
            stim_inputs = stim_inputs + torch.randn(stim_inputs.shape, device=device, dtype=torch.float)/5

        assert(stim_inputs.shape[3]==1) # no multiple screens
        # multimodal transformer
        stim_token_pad = stim_inputs.shape[1]
        stim_embedding_pad = stim_inputs.shape[2]
        rule_token_pad = rule_inputs.shape[1]
        rule_embedding_pad = rule_inputs.shape[2]
        # right padding last dim (embed_pad) and 2nd last dim (token_pad)
        train_rules_pad = torch.nn.functional.pad(rule_inputs,(0,stim_embedding_pad,0,stim_token_pad),'constant')
        train_stim_pad = torch.nn.functional.pad(stim_inputs[:,:,:,0],(rule_embedding_pad,0,rule_token_pad,0),'constant')
        # merge zeropadded inputs
        combined_inputs = train_rules_pad + train_stim_pad 

        embedding = self.w_embed(combined_inputs)

        ####
        # initialize RNN state 
        seq_length = rule_inputs.size()[1]
        batch_size = rule_inputs.size()[0]
        # rule transformer block
        transformer_out = checkpoint_sequential(self.blocks, segments = len(self.blocks), input = embedding)

        outputs = self.w_out(transformer_out[:,-1,:])
        return outputs, outputs # only want the last output from seq data

class TransformerBlock2(torch.nn.Module):
    """
    Transformer block
    """
    def __init__(self,
                 embedding_dim,
                 n_tokens,
                 positional_encoding='absolute',
                 nhead=1,
                 dropout=0,
                 learning_rate=0.0001,
                 lossfunc='CrossEntropy'):
        super(TransformerBlock2,self).__init__()

        # Define general parameters
        self.nhead = nhead
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_embed = torch.nn.Dropout(p=dropout)
        self.positional_encoding = positional_encoding
        
        # positional encoding
        if positional_encoding=='absolute':
            self.pe = PositionalEncoding(self.embedding_dim, max_len=n_tokens)
            self.selfattention = torch.nn.MultiheadAttention(self.embedding_dim,nhead,dropout,batch_first=True)
        elif positional_encoding=='relative':
            self.selfattention = RelativeGlobalAttention(self.embedding_dim, nhead, dropout=dropout, max_len=n_tokens)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim,self.embedding_dim*4),
            torch.nn.GELU(),
            torch.nn.Linear(self.embedding_dim*4,self.embedding_dim),
            torch.nn.GELU()
        )

        # layer norm; 1st is after attention (embedding dim); 2nd is after RNN 
        self.layernorm0 = torch.nn.LayerNorm(self.embedding_dim)
        self.layernorm1 = torch.nn.LayerNorm(self.embedding_dim)

    def forward(self, embedding, noise=False, dropout=False):
        device = embedding.device
        #Add noise to inputs
        if noise:
            embedding = embedding + torch.randn(embedding.shape, device=device, dtype=torch.float)/5

        ####
        # transformer block
        if self.positional_encoding in ['relative']:
            attn_outputs = self.selfattention(embedding)
        else:
            embedding = self.pe(embedding) # positional encoding
            embedding = self.dropout_embed(embedding)
            attn_outputs, attn_out_weights = self.selfattention(embedding, embedding, embedding, need_weights=False)
        attn_outputs = self.layernorm0(attn_outputs+embedding) # w resid connection
        transformer_out = self.mlp(attn_outputs)
        transformer_out = self.layernorm1(transformer_out+attn_outputs) # w resid connection

        return transformer_out
