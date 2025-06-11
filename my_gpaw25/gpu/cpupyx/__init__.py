def __getattr__(attr):
    if attr == 'scipy':
        import my_gpaw25.gpu.cpupyx.scipy as scipy
        return scipy
    raise AttributeError(attr)
