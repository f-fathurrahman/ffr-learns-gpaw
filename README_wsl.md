

Need to manually install Libxc
Probably need to set LD_LIBRARY_PATH

Easier using conda env, need to install libxc-dev first.
Now installed gpaw version is 25.x.x
my_gpaw is using 23.x.x version.

```bash
export GPAW_SETUP_PATH=$PWD/GPAW_SETUPS
```

Using `ipdb` instead of `pdb`:
```bash
pip install ipdb
export PYTHONBREAKPOINT="ipdb.set_trace"
```
