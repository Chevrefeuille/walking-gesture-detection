import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def parse_trajectory(file_path):
    """
    Parse the trajectory contained in the given file
        input: file path as a string
        output: lists containing the time and (x, y) couples
    """
    time, coords = [], []
    print(file_path)
    with open(file_path) as f:
        n = int(f.readline())
        for _ in range(n):
            line = f.readline().split(";")
            time.append(datetime.fromtimestamp(
                    int(float(line[0])/1000)
                ).strftime('%Y-%m-%d %H:%M:%S')
            )
            coords.append((float(line[1]), float(line[2])))
    return time, coords
        
            
            

pathlist = Path("./crowd").glob('*.csv')
for path in pathlist:
    path_in_str = str(path)
    time, coords = parse_trajectory("crowd/path_15092800.csv")
    print(time)
    plt.plot([c[0] for c in coords], [c[1] for c in coords])
    plt.show()
    break
    