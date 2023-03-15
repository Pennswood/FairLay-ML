import subprocess
timeout=90
max_iter=50
save_model = True
subprocess.call(['./m_mutation_testing-updates.sh', f"{timeout}", f"{save_model}", f"{max_iter}"])


import pandas as pd

classifiers = ["LR", "RF", "SV", "DT"]
test = [("census", "gender"), ("census", "race"), ("credit", "gender"), ("bank", "age"), ("compas", "gender"), ("compas", "race")]
