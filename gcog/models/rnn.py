import numpy as np
import torch
import gcog.task.config as config

class RNN(torch.nn.RNN):
    """
    Vanilla feedforward rnn 
    """

    def __init__(self,
                 num_rule_inputs=config.RULE_ARRAY_LENGTH,
                 num_stim_inputs=(config.GRIDSIZE_X*config.GRIDSIZE_Y+1)*config.STIMULI_ARRAY_LENGTH,
                 num_hidden=512,
                 num_motor_decision_outputs=138,
                 num_hidden_layers=1,
                 learning_rate=0.0001,
                 thresh=0.0,
                 lossfunc='CrossEntropy'):

        # Define general parameters
        self.num_rule_inputs = num_rule_inputs
        self.num_stim_inputs = num_stim_inputs 
        self.num_inputs = self.num_rule_inputs + self.num_stim_inputs
        self.num_hidden = num_hidden
        self.num_motor_decision_outputs = num_motor_decision_outputs
        self.num_hidden_layers = num_hidden_layers
        
        # Define network architectural parameters
        super(RNN,self).__init__(self.num_inputs,
                                 num_hidden,
                                 num_layers=num_hidden_layers,
                                 nonlinearity='relu',
                                 batch_first=True,
                                 dropout=0)

        # for output of RNN (before w_otu)
        self.layernorm = torch.nn.LayerNorm(self.num_hidden)
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
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        
    def forward(self, rule_inputs, stim_inputs, noise=False,dropout=False):
        """
        Run a forward pass of a trial by input_elements matrix
        For each time window, pass each 
        rule_inputs (Tensor): batch x seq_length x rule_vec
        stim_inputs (Tensor): batch x seq_length x stim_vec x time
        """
        assert(rule_inputs.device==stim_inputs.device)
        device = rule_inputs.device
        #Add noise to inputs
        if noise:
            rule_inputs = rule_inputs + torch.randn(rule_inputs.shape, device=device, dtype=torch.float)/5
            stim_inputs = stim_inputs + torch.randn(stim_inputs.shape, device=device, dtype=torch.float)/5

        assert(stim_inputs.shape[3]==1) # only one stimulus image
        # initialize RNN state 
        seq_length = rule_inputs.size()[1]
        batch_size = rule_inputs.size()[0]
        hn = torch.zeros(self.num_hidden_layers,
                         batch_size,
                         self.num_hidden, 
                         device=device)
        # Number of inputs depends on the length of the rule sequence
        inputs = torch.zeros(batch_size,seq_length,rule_inputs.shape[2]+stim_inputs.shape[1]*stim_inputs.shape[2],device=device)
        for i in range(seq_length):
            rule_inputs_flat = torch.squeeze(rule_inputs[:,i,:])
            stim_inputs_flat = stim_inputs.reshape(stim_inputs[:,:,:,0].shape[0],-1)
            concat = torch.concat((rule_inputs_flat,stim_inputs_flat),dim=1)
            inputs[:,i,:] = concat

        outputs, hn = super(RNN,self).forward(inputs,hn) 
        hn = self.layernorm(hn)
        outputs = torch.squeeze(self.layernorm(outputs))
        outputs = self.w_out(hn[-1]) #1 x batch x output

        return outputs, hn # only want the last output from seq data

class GRU(torch.nn.GRU):
    """
    Vanilla GRU network 
    """

    def __init__(self,
                 num_rule_inputs=config.RULE_ARRAY_LENGTH,
                 num_stim_inputs=(config.GRIDSIZE_X*config.GRIDSIZE_Y+1)*(len(config.ALL_COLORS)+len(config.ALL_SHAPES)+1), # (10 * 10+1) * (26 shapes + 10 colors + EOS)
                 num_hidden=512,
                 num_motor_decision_outputs=138,
                 num_hidden_layers=1,
                 learning_rate=0.0001,
                 lossfunc='CrossEntropy'):

        # Define general parameters
        self.num_rule_inputs = num_rule_inputs
        self.num_stim_inputs =  num_stim_inputs
        self.num_inputs = num_rule_inputs + num_stim_inputs
        self.num_hidden = num_hidden
        self.num_motor_decision_outputs = num_motor_decision_outputs
        self.num_hidden_layers = num_hidden_layers
        
        # Define network architectural parameters
        super(GRU,self).__init__(self.num_inputs,
                                 num_hidden,
                                 num_layers=num_hidden_layers,
                                 batch_first=True,
                                 dropout=0)

        # for output of GRU (before w_otu)
        self.layernorm = torch.nn.LayerNorm(self.num_hidden)
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
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        
    def forward(self, rule_inputs, stim_inputs, noise=False,dropout=False):
        """
        Run a forward pass of a trial by input_elements matrix
        For each time window, pass each 
        rule_inputs (Tensor): batch x seq_length x rule_vec
        stim_inputs (Tensor): batch x seq_length x stim_vec x time
        """
        assert(rule_inputs.device==stim_inputs.device)
        device = rule_inputs.device
        #Add noise to inputs
        if noise:
            rule_inputs = rule_inputs + torch.randn(rule_inputs.shape, device=device, dtype=torch.float)/5
            stim_inputs = stim_inputs + torch.randn(stim_inputs.shape, device=device, dtype=torch.float)/5

        assert(stim_inputs.shape[3]==1) # only one stimulus image
        # initialize RNN state 
        seq_length = rule_inputs.size()[1]
        batch_size = rule_inputs.size()[0]
        hn = torch.zeros(self.num_hidden_layers,
                         batch_size,
                         self.num_hidden, 
                         device=device)
        # Number of inputs depends on the length of the rule sequence
        inputs = torch.zeros(batch_size,seq_length,rule_inputs.shape[2]+stim_inputs.shape[1]*stim_inputs.shape[2],device=device)
        for i in range(seq_length):
            rule_inputs_flat = torch.squeeze(rule_inputs[:,i,:])
            stim_inputs_flat = stim_inputs.reshape(stim_inputs[:,:,:,0].shape[0],-1)
            concat = torch.concat((rule_inputs_flat,stim_inputs_flat),dim=1)
            inputs[:,i,:] = concat

        outputs, hn = super(GRU,self).forward(inputs,hn) 
        hn = self.layernorm(hn)
        outputs = torch.squeeze(self.layernorm(outputs))
        outputs = self.w_out(hn[-1]) #1 x batch x output

        return outputs, hn # only want the last output from seq data
