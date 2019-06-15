import os
import re

class MWIS():
    """Class for solving weighted maximum independent sets.
    Using methods in paper: Optimisation of unweighted/weighted maximum independent sets and minimum vertex covers
    github: https://github.com/darrenstrash/open-pls.git
    """

    def __init__(self, name, path='inputs/graph'):
        self.name = name
        self.graph_file = os.path.join(path, self.name+'.graph')
        self.weight_file = os.path.join(path, self.name+'.graph.weights')
        self.out_file = os.path.join(path, self.name+'.sol')

    def local_search(self):
        """Find weighted maximum independent sets in given graph
        """
        command = './open-pls-1.0/bin/pls --algorithm=mwis --input-file=' + self.graph_file + ' --weighted --use-weight-file --timeout=1 --random-seed=0 > ' + self.out_file
        os.system(command)
        # read results
        sol_pattern = re.compile("""best-solution   :(( \d+)+)""")
        with open(self.out_file, 'r') as f:
            content = f.read()
        sol_match = re.search(sol_pattern, content)
        if sol_match:
            node_list = [int(x) for x in sol_match.group(1).split()]
            return node_list
        else:
            print("ERROR!: no mwis solution found")
            return -1

    def write_graph(self, str_list, str_weights):
        """Write graph into file
        """
        with open(self.graph_file, 'w') as f:
            f.writelines(str_list)
        with open(self.weight_file, 'w') as f:
            f.writelines(str_weights)

def main():
    mwis = MWIS('test-my')
    results = mwis.local_search()
    print(results)

if __name__ == '__main__':
    main()