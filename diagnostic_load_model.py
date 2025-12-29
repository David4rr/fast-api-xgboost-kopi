import joblib, os, traceback
p='saved/model.pkl'
print('Path:', p)
print('Exists:', os.path.exists(p))
if os.path.exists(p):
    try:
        print('Size (bytes):', os.path.getsize(p))
        obj = joblib.load(p)
        print('Loaded type:', type(obj))
        try:
            if isinstance(obj, dict):
                print('Dict keys:', list(obj.keys()))
            else:
                print('Object repr:', repr(obj)[:400])
                if hasattr(obj, '__dict__'):
                    keys = list(obj.__dict__.keys())
                    print('Attr keys sample:', keys[:50])
        except Exception as e:
            print('Error introspecting object:', e)
    except Exception as e:
        print('Error loading model:')
        traceback.print_exc()