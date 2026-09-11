import subprocess
import argparse


def print_sata_sim_banner():
    banner = r"""
   _____      _______                   _____ _____ __  __ 
  / ____|  /\|__   __|/\               / ____|_   _|  \/  |
 | (___   /  \  | |  /  \     ______  | (___   | | | \  / |
  \___ \ / /\ \ | | / /\ \   |______|  \___ \  | | | |\/| |
  ____) / ____ \| |/ ____ \            ____) |_| |_| |  | |
 |_____/_/    \_\_/_/    \_\          |_____/|_____|_|  |_|
                                                                                                               
    """
    print(banner)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SATA simulation utilities.")
    parser.add_argument("-c", "--config", default="sata-config.yaml", help="Path to the configuration YAML file.",)
    parser.add_argument("-w", "--workload", default="workload.yaml", help="Path to the workload YAML file.",)
    args = parser.parse_args()

    print_sata_sim_banner()
    subprocess.run(["python3", "comp-utils.py", "-c", args.config, "-w", args.workload], check=True)
    subprocess.run(["python3", "mem-utils.py", "-c", args.config, "-w", args.workload], check=True)
    subprocess.run(["python3", "cycle-utils.py", "-c", args.config, "-w", args.workload], check=True)
    subprocess.run(["python3", "energy-cal.py", "-c", args.config, "-w", args.workload], check=True)
