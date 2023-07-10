from my_gpaw.atom.configurations import parameters
from my_gpaw.atom.generator import Generator

xcname = "LDA"
symbol = "Si"
par = parameters[symbol]

filename = symbol + '.' + xcname + '.xml'
g = Generator(symbol, xcname, scalarrel=True, nofiles=True)
g.run(exx=True, logderiv=False, write_xml=False, **par)
