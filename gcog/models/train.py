import torch

def train(network, rule_inputs, stim_inputs, targets, dropout=False):
    """Train network"""
    network.train()
    network.zero_grad()
    network.optimizer.zero_grad()

    outputs, hidden = network.forward(rule_inputs,
                                      stim_inputs,
                                      noise=False,
                                      dropout=dropout)

    # Calculate loss
    loss = network.lossfunc(outputs,targets)
    loss = torch.mean(loss)

    # Backprop and update weights
    loss.backward()

    max_norm = 1.0

    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm, norm_type=2)

    if hasattr(network, 'scheduler'):
        network.scheduler.step(loss)
        network.optimizer.step()
    else:
        network.optimizer.step() # Update parameters using optimizer
    
    return outputs, loss.item()


