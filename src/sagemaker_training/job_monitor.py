import boto3

from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

from sagemaker_training import (
    logging_config,
)

logger = logging_config.get_logger()
SAGEMAKER_TRAINING_JOB_LOG_GROUP = '/aws/sagemaker/TrainingJobs'
STUCK_JOB_MONITOR_SLEEP_TIME = 300

class JobMonitor:
    def __init__(self, region, missing_cw_log_output_limit_mins, filter_keywords = []):
        self.region=region
        session = boto3.Session(region_name=region)
        self.logs_client = session.client('logs', region)
        self.missing_cw_log_output_limit_mins = missing_cw_log_output_limit_mins
        self.log_group_name = SAGEMAKER_TRAINING_JOB_LOG_GROUP
        self.filter_pattern = " ".join([f'"{word}"' for word in filter_keywords])
        self.current_time = str(datetime.now(timezone.utc)).split('.', maxsplit=1)[0]
        self.sleep_time_secs = STUCK_JOB_MONITOR_SLEEP_TIME

    # pylint: disable=too-many-arguments
    def get_log_events_from_stream(self, log_group, log_stream_name, start_time, end_time, filter_pattern = ""):
        start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

        # Convert datetime to milliseconds since the epoch
        epoch_time = datetime(1970, 1, 1)
        start_timestamp = int((start_time - epoch_time).total_seconds() * 1000)
        end_timestamp  = int((end_time - epoch_time).total_seconds() * 1000)

        next_token = None
        all_events = []

        while True:
            args = {
                "logGroupName": log_group,
                "logStreamNames": [log_stream_name],
                "startTime": start_timestamp,
                "endTime": end_timestamp,
                "filterPattern": filter_pattern
            }
            
            if next_token:
                args["nextToken"] = next_token

            response = self.logs_client.filter_log_events(**args)
            all_events.extend(response.get('events', []))

            next_token = response.get('nextToken')
            if not next_token:
                break
        return all_events

    # Usage 
    # if detect_stuck_training_job(job_log_stream_name):
    #   logger.info("Training job is stuck. Exiting...")
    #   sys.exit(1)
    def detect_stuck_training_job(self, job_log_stream_name):
        start_time_epoch_query = datetime.strptime(self.current_time, '%Y-%m-%d %H:%M:%S')
        start_time_epoch_query = str(start_time_epoch_query - relativedelta(minutes = self.missing_cw_log_output_limit_mins))
        logs = self.get_log_events_from_stream(self.log_group_name, job_log_stream_name, start_time_epoch_query, self.current_time, filter_pattern=self.filter_pattern)
        logger.info(f"Number of expected log extry discovered in the past {self.missing_cw_log_output_limit_mins} minutes: {len(logs)}")
        # if there is no expected log entry in the past self.missing_cw_log_output_limit_mins, then the training job is stuck
        return len(logs) == 0    