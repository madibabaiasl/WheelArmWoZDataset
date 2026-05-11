# # import hydra
# # from openteach.components import TeleOperator

# # @hydra.main(version_base = '1.2', config_path = 'configs', config_name = 'teleop')
# # def main(configs):
# #     teleop = TeleOperator(configs)
# #     processes = teleop.get_processes()

# #     for process in processes:
# #         process.start()

# #     for process in processes:
# #         process.join()

# # if __name__ == '__main__':
# #     main()

# import hydra
# from openteach.components import TeleOperator

# @hydra.main(version_base='1.2', config_path='configs', config_name='teleop')
# def main(configs):
#     teleop = TeleOperator(configs)
#     processes = teleop.get_processes()
    
#     for i, process in enumerate(processes):
#         print(f"[TELEOP_START] idx={i} name={process.name} before_start_pid={process.pid}")

#         print(f"[TELEOP_STARTED] idx={i} name={process.name} pid={process.pid}")

#         try:
#             for process in processes:
#                 process.start()
#                 print(f"[TELEOP_STARTED] idx={i} name={process.name} pid={process.pid}")

#             for process in processes:
#                 process.join()

#         except KeyboardInterrupt:
#             print("Stopping child processes...")

#         finally:
#             for process in processes:
#                 if process.is_alive():
#                     process.terminate()

#             for process in processes:
#                 process.join(timeout=2)

#             for process in processes:
#                 if process.is_alive():
#                     process.kill()
# if __name__ == '__main__':
#     main()

import hydra
from openteach.components import TeleOperator

@hydra.main(version_base='1.2', config_path='configs', config_name='teleop')
def main(configs):
    teleop = TeleOperator(configs)
    processes = teleop.get_processes()

    try:
        for i, proc in enumerate(processes):
            print(f"[TELEOP_BEFORE_START] idx={i} name={proc.name} pid={proc.pid}")
            proc.start()
            print(f"[TELEOP_AFTER_START]  idx={i} name={proc.name} pid={proc.pid}")

        for i, proc in enumerate(processes):
            print(f"[TELEOP_JOIN_WAIT]    idx={i} name={proc.name} pid={proc.pid}")
            proc.join()
            print(f"[TELEOP_JOIN_DONE]    idx={i} name={proc.name} pid={proc.pid} exitcode={proc.exitcode}")

    except KeyboardInterrupt:
        print("Stopping child processes...")

    finally:
        for i, proc in enumerate(processes):
            if proc.is_alive():
                print(f"[TELEOP_TERMINATE]    idx={i} name={proc.name} pid={proc.pid}")
                proc.terminate()

        for i, proc in enumerate(processes):
            proc.join(timeout=2)

        for i, proc in enumerate(processes):
            if proc.is_alive():
                print(f"[TELEOP_KILL]         idx={i} name={proc.name} pid={proc.pid}")
                proc.kill()

if __name__ == '__main__':
    main()