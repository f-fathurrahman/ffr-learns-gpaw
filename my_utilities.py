
def typeof(obj):
    typ_str = str(type(obj)).replace("<", "").replace(">", "")
    # typ_str = str(type(self)).replace("<", "").replace(">", "")
    return typ_str