import hydra
from openteach.components import OakCameras

@hydra.main(version_base = '1.2', config_path = 'configs', config_name = 'camera')
def main(configs):
    oak_mgr = OakCameras(configs)
    processes = oak_mgr.get_processes()

    for process in processes:
        process.start()

    for process in processes:
        process.join()

if __name__ == '__main__':
    main()
