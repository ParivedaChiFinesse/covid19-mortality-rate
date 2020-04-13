import pymc3 as pm

def train_model(mod):

    with mod:
        trace_default = pm.sample()
    return trace_default