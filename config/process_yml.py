import yaml
import os
import sys


def create_temp_yaml(src_yml_path, dst_yml_path="temp.yml"):
    f = open(src_yml_path, 'r')
    yml_dict = yaml.load(f, yaml.Loader)
    f.close()

    for key in yml_dict.keys():
        if "DIR" in key or key == "REID_MODEL":
            if key == "MAIN_DIR":
                continue
            yml_dict[key] = os.path.join(yml_dict["MAIN_DIR"], yml_dict[key])
    
    w = open(dst_yml_path, 'w')
    yaml.dump(yml_dict, w, yaml.Dumper)
    w.close()

if __name__ == '__main__':
    create_temp_yaml(sys.argv[1], sys.argv[2])
