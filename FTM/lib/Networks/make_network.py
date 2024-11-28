import importlib as imp

def make_network(opt):
    if opt.model == 'Vector':
        network = imp.machinery.SourceFileLoader(
            opt.vec_module, opt.vec_path).load_module().FVecNet()
    
    else:
        network = imp.machinery.SourceFileLoader(
            opt.FP_module, opt.FP_path).load_module().ContNetwork
        network = network(opt)

    return network
