
from pipeline_prefect import flow_all

if __name__ == "__main__":
    flow_all.serve(
        name="ml-pipeline-all",
        cron="0 8 * * *",
    )
