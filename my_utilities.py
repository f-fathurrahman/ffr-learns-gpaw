
def typeof(obj):
    typ_str = str(type(obj)).replace("<", "").replace(">", "")
    return typ_str