import json
from types import SimpleNamespace as Namespace

class SimpleNamespace (object):
    def __init__ (self, **kwargs):
        self.__dict__.update(kwargs)
    def __repr__ (self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))
    def __eq__ (self, other):
        return self.__dict__ == other.__dict__


def main():
    filename = "D:/Participants/NRI-Kids_videos-tree.json"
    with open(filename, "r") as f:
        line = f.readlines()[0]
        all_files = json.loads(line, object_hook=lambda d: Namespace(**d))
    files = all_files.__dict__["Participant #2"].videos

    for week in files.__dict__:
        try:
            for name in files.__dict__[week].__dict__["."]:
                print(week + "/" + name)
        except:
            pass
        for day in files.__dict__[week].__dict__.keys():
            try:
                for name in files.__dict__[week].__dict__[day].__dict__["."]:
                    print(week + "/" + day + "/" + name)
            except:
                pass

if __name__ == "__main__":
    main()
