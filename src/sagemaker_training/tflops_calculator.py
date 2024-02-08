from sagemaker_training import (
    environment,
    logging_config,
)

logger = logging_config.get_logger()
class TflopsCalculator:
    def __init__(self):
        pass

    def get_model_tflops(self, num_params, iter_time, tokens_per_gpu, attention=0):
        # Calculated according to https://arxiv.org/pdf/2204.02311.pdf
        model_tflops = ((6 * num_params + 12 * attention) * (tokens_per_gpu / iter_time)) / 1e12
        return model_tflops

    def compute_mfu(self, num_params, iter_time, tokens_per_gpu, world_size, attention=0):
        R = 312*1e12 / (6*num_params)
        tokens_per_second = (tokens_per_gpu * world_size)/iter_time
        env = environment.Environment()
        if env.master_hostname == env.current_host:
            logger.info(f"R: {R}")
            logger.info(f"Tokens per second: {tokens_per_second}")
            logger.info(f"Iter time: {iter_time}")
            logger.info(f"Tokens per GPU: {tokens_per_gpu}")
            logger.info(f"Num params: {num_params}")
            logger.info(f"World size: {world_size}")
            logger.info(f"Attention: {attention}")
            logger.info(f"MFU: {(tokens_per_second / R) * 100}")
            logger.info(f"Model TFLOPS/GPU: {self.get_model_tflops(num_params, iter_time, tokens_per_gpu)}")
            logger.info(f"Model TFLOPS/GPU (with attention): {self.get_model_tflops(num_params, iter_time, tokens_per_gpu, attention)}")

    def log_tflops(self, num_params, iter_time, tokens_per_gpu, attention=0):
        model_tflops = self.get_model_tflops(num_params, iter_time, tokens_per_gpu, attention)
        env = environment.Environment()
        if env.master_hostname == env.current_host:
            logger.info(f"Num params: {num_params}")
            logger.info(f"Iter time: {iter_time}")
            logger.info(f"Tokens per GPU: {tokens_per_gpu}")
            logger.info(f"Model TFLOPS/GPU: {model_tflops}")