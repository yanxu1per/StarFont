import torch
import torchvision.models
import hiddenlayer as hl
import model
import pdb
g=model.Generator(64, 10, 6)
d=model.Discriminator(128, 64, 10, 6)
#x=hl.build_graph(d, torch.zeros([16,1, 128, 128]))

x=hl.build_graph(g, (torch.zeros([1,1, 128, 128]),torch.zeros([1,10])))


pdb.set_trace()