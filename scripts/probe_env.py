import importlib

mods = ['scipy', 'sklearn', 'causallearn', 'torch', 'networkx',
        'pandas', 'matplotlib', 'castle', 'tigramite', 'ruptures', 'avici']
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f'{m}: {getattr(mod, "__version__", "?")}')
    except Exception:
        print(f'{m}: MISSING')

try:
    import torch
    print('cuda available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('device:', torch.cuda.get_device_name(0))
except Exception as e:
    print('torch cuda check failed:', e)
