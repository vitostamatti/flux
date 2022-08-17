
import inspect
import sys
import json


def _isnotebook():
    import __main__ as main
    return not hasattr(main, '__file__')


def _getfile(object, _old_getfile=inspect.getfile):
    if not inspect.isclass(object):
        return _old_getfile(object)
    
    if hasattr(object, '__module__'):
        object_ = sys.modules.get(object.__module__)
        if hasattr(object_, '__file__'):
            return object_.__file__
    
    for name, member in inspect.getmembers(object):
        if inspect.isfunction(member) and object.__qualname__ + '.' + member.__name__ == member.__qualname__:
            return inspect.getfile(member)
    else:
        raise TypeError('Source for {!r} not found'.format(object))


def _get_notebook_imports(filename):
    ""
    input_file = filename
    f = open(input_file, 'r') 
    j = json.load(f)
    of = {}
    for i,cell in enumerate(j["cells"]):
        of[i]=[]  
        for line in cell["source"]:
            of[i].append(line)

    import_lines = []
    for c in of:
        for l in of[c]:
            if (l.startswith("import ")) or (l.startswith("from ") and "import" in l):
                if not l.endswith('\n'):
                    l = l+"\n"
                import_lines.append(l)

    return "".join(import_lines)
