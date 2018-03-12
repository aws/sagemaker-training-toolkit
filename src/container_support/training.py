import traceback
from container_support import TrainingEnvironment


class Trainer(object):
    @classmethod
    def start(cls):
        base_dir = None
        try:
            env = TrainingEnvironment()
            env.start_metrics_if_enabled()
            base_dir = env.base_dir

            env.pip_install_requirements()

            fw = TrainingEnvironment.load_framework()
            fw.train()
            env.write_success_file()
        except Exception as e:
            trc = traceback.format_exc()
            message = 'uncaught exception during training: {}\n{}\n'.format(e, trc)
            TrainingEnvironment.write_failure_file(message, base_dir)
            raise e
