import json
from types import SimpleNamespace as Namespace


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