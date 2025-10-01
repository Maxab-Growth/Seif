import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)



def get_secret(secret_name):
    import boto3
    import base64
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except Exception as e:
        raise e
    else:
        if "SecretString" in get_secret_value_response:
            return get_secret_value_response["SecretString"]
        else:
            return base64.b64decode(get_secret_value_response["SecretBinary"])


def initialize_env():
    import json
    from pathlib import Path
    import os
    import boto3

    # Loads the secrets from Secrets Manager
    dwh_reader_secret = json.loads(get_secret("prod/db/datawarehouse/metabase"))
    dwh_writer_secret = json.loads(get_secret("prod/db/datawarehouse/sagemaker"))
    snowflake_secret = json.loads(get_secret("Snowflake-sagemaker"))
    snowflake_users_secret = json.loads(get_secret("snowflake/users"))
    maxroute_dwh_secret= json.loads(get_secret("prod/rds/maxroute"))
    morocco_metabase_access= json.loads(get_secret("prod/metabase/mohamed_ashraf/user"))
    zamabia_dwh_writer_secret=json.loads(get_secret("prod/db/zambia/sagemaker"))
    zamabia_dwh_reader_secret=json.loads(get_secret("prod/db/zambia/reader"))
    kenya_dwh_writer_secret=json.loads(get_secret("prod/db/kenya/sagemaker"))
    kenya_dwh_reader_secret=json.loads(get_secret("prod/db/kenya/reader"))
    rwanda_dwh_writer_secret=json.loads(get_secret("prod/db/rwanda/sagemaker"))
    rwanda_dwh_reader_secret=json.loads(get_secret("prod/db/rwanda/reader"))
    tanzania_dwh_writer_secret=json.loads(get_secret("prod/db/tanzania/sagemaker"))
    tanzania_dwh_reader_secret=json.loads(get_secret("prod/db/tanzania/reader"))
    snowflake_secret = json.loads(get_secret("Snowflake-sagemaker"))
    mongo_creds = json.loads(get_secret("prod/db/mongodb/sagemaker"))
    maintained_schemas_buckets = json.loads(get_secret("prod/s3_buckets_maintained_schemas"))
    shamselkheir_odoo_secret = json.loads(get_secret("prod/odoo/shamselkheir"))
    three_cx_egypt_secret = json.loads(get_secret('prod/3cx/postgres'))
    snowflake_dbt_maxab = json.loads(get_secret("prod/dbt/dbt_maxab"))
    maxsupport_secret = json.loads(get_secret("prod/db/maxsupport/writer"))
    fintech_service_account = json.loads(get_secret("prod/fintechServiceEmail/credentials"))

    # RETRIEVING MAINTAINED SCHEMAS S3 BUCKETS NAMES
    os.environ["EGYPT_MAINTAINED_BUCKET"] = maintained_schemas_buckets["egypt_maintained"]
    os.environ["KENYA_MAINTAINED_BUCKET"] = maintained_schemas_buckets["kenya_maintained"]
    os.environ["MOROCCO_MAINTAINED_BUCKET"] = maintained_schemas_buckets["morocco_maintained"]
    os.environ["RWANDA_MAINTAINED_BUCKET"] = maintained_schemas_buckets["rwanda_maintained"]
    os.environ["TANZANIA_MAINTAINED_BUCKET"] = maintained_schemas_buckets["tanzania_maintained"]
    
    # RETRIEVING METABASE SECRET
    metabase_secret = json.loads(get_secret("prod/metabase/maxab_config"))
    os.environ["EGYPT_METABASE_USERNAME"] = metabase_secret["metabase_user"]
    os.environ["EGYPT_METABASE_PASSWORD"] = metabase_secret["metabase_password"]
    os.environ["EGYPT_METABASE_URL"] = metabase_secret["metabase_egypt_site"]
    
    os.environ["AFRICA_METABASE_USERNAME"] = metabase_secret["metabase_user"]
    os.environ["AFRICA_METABASE_PASSWORD"] = metabase_secret["metabase_password"]
    os.environ["AFRICA_METABASE_URL"] = metabase_secret["metabase_morocco_site"]

    # ########## countries DWH credentials ##########
    #Egypt
    os.environ["EGYPT_DWH_READER_HOST"] = dwh_reader_secret["host"]
    os.environ["EGYPT_DWH_READER_NAME"] = dwh_reader_secret["dbname"]
    os.environ["EGYPT_DWH_READER_USER_NAME"] = dwh_reader_secret["username"]
    os.environ["EGYPT_DWH_READER_PASSWORD"] = dwh_reader_secret["password"]
    
    os.environ["EGYPT_DWH_WRITER_HOST"] = dwh_writer_secret["host"]
    os.environ["EGYPT_DWH_WRITER_NAME"] = dwh_writer_secret["dbname"]
    os.environ["EGYPT_DWH_WRITER_USER_NAME"] = dwh_writer_secret["username"]
    os.environ["EGYPT_DWH_WRITER_PASSWORD"] = dwh_writer_secret["password"]
    
    #Morocco
    os.environ["MOROCCO_DWH_READER_HOST"] = dwh_reader_secret["host"]
    os.environ["MOROCCO_DWH_READER_NAME"] = 'morocco'
    os.environ["MOROCCO_DWH_READER_USER_NAME"] = dwh_reader_secret["username"]
    os.environ["MOROCCO_DWH_READER_PASSWORD"] = dwh_reader_secret["password"]
    
    os.environ["MOROCCO_DWH_WRITER_HOST"] = dwh_writer_secret["host"]
    os.environ["MOROCCO_DWH_WRITER_NAME"] = 'morocco'
    os.environ["MOROCCO_DWH_WRITER_USER_NAME"] = dwh_writer_secret["username"]
    os.environ["MOROCCO_DWH_WRITER_PASSWORD"] = dwh_writer_secret["password"]
    
    #Zambia
    os.environ["ZAMBIA_DWH_READER_HOST"] = zamabia_dwh_reader_secret["host"]
    os.environ["ZAMBIA_DWH_READER_NAME"] = zamabia_dwh_reader_secret["dbname"]
    os.environ["ZAMBIA_DWH_READER_USER_NAME"] = zamabia_dwh_reader_secret["username"]
    os.environ["ZAMBIA_DWH_READER_PASSWORD"] = zamabia_dwh_reader_secret["password"]
    
    os.environ["ZAMBIA_DWH_WRITER_HOST"] = zamabia_dwh_writer_secret["host"]
    os.environ["ZAMBIA_DWH_WRITER_NAME"] = zamabia_dwh_writer_secret["dbname"]
    os.environ["ZAMBIA_DWH_WRITER_USER_NAME"] = zamabia_dwh_writer_secret["username"]
    os.environ["ZAMBIA_DWH_WRITER_PASSWORD"] = zamabia_dwh_writer_secret["password"]
    
    #Kenya
    os.environ["KENYA_DWH_READER_HOST"] = kenya_dwh_reader_secret["host"]
    os.environ["KENYA_DWH_READER_NAME"] = kenya_dwh_reader_secret["dbname"]
    os.environ["KENYA_DWH_READER_USER_NAME"] = kenya_dwh_reader_secret["username"]
    os.environ["KENYA_DWH_READER_PASSWORD"] = kenya_dwh_reader_secret["password"]
    
    os.environ["KENYA_DWH_WRITER_HOST"] = kenya_dwh_writer_secret["host"]
    os.environ["KENYA_DWH_WRITER_NAME"] = kenya_dwh_writer_secret["dbname"]
    os.environ["KENYA_DWH_WRITER_USER_NAME"] = kenya_dwh_writer_secret["username"]
    os.environ["KENYA_DWH_WRITER_PASSWORD"] = kenya_dwh_writer_secret["password"]

    #Rwanda
    os.environ["RWANDA_DWH_READER_HOST"] = rwanda_dwh_reader_secret["host"]
    os.environ["RWANDA_DWH_READER_NAME"] = rwanda_dwh_reader_secret["dbname"]
    os.environ["RWANDA_DWH_READER_USER_NAME"] = rwanda_dwh_reader_secret["username"]
    os.environ["RWANDA_DWH_READER_PASSWORD"] = rwanda_dwh_reader_secret["password"]
    
    os.environ["RWANDA_DWH_WRITER_HOST"] = rwanda_dwh_writer_secret["host"]
    os.environ["RWANDA_DWH_WRITER_NAME"] = rwanda_dwh_writer_secret["dbname"]
    os.environ["RWANDA_DWH_WRITER_USER_NAME"] = rwanda_dwh_writer_secret["username"]
    os.environ["RWANDA_DWH_WRITER_PASSWORD"] = rwanda_dwh_writer_secret["password"] 

    #Tanzania
    os.environ["TANZANIA_DWH_READER_HOST"] = tanzania_dwh_reader_secret["host"]
    os.environ["TANZANIA_DWH_READER_NAME"] = tanzania_dwh_reader_secret["dbname"]
    os.environ["TANZANIA_DWH_READER_USER_NAME"] = tanzania_dwh_reader_secret["username"]
    os.environ["TANZANIA_DWH_READER_PASSWORD"] = tanzania_dwh_reader_secret["password"]
    
    os.environ["TANZANIA_DWH_WRITER_HOST"] = tanzania_dwh_writer_secret["host"]
    os.environ["TANZANIA_DWH_WRITER_NAME"] = tanzania_dwh_writer_secret["dbname"]
    os.environ["TANZANIA_DWH_WRITER_USER_NAME"] = tanzania_dwh_writer_secret["username"]
    os.environ["TANZANIA_DWH_WRITER_PASSWORD"] = tanzania_dwh_writer_secret["password"] 

    # ########## countries SNOWFLAKE credentials ##########
    os.environ["SNOWFLAKE_USERNAME"] = snowflake_secret["username"]
    os.environ["SNOWFLAKE_PASSWORD"] = snowflake_secret["password"]
    os.environ["SNOWFLAKE_ACCOUNT"] = snowflake_secret["account"]
    os.environ["SNOWFLAKE_AIRFLOW_WAREHOUSE"] = snowflake_secret["airflow_scripts_main_warehouse"]
    os.environ["EGYPT_SNOWFLAKE_DATABASE"] = snowflake_secret["database"]
    os.environ["RWANDA_SNOWFLAKE_DATABASE"] = snowflake_secret["rwanda_database"]
    os.environ["MOROCCO_SNOWFLAKE_DATABASE"] = snowflake_secret["morocco_database"]
    os.environ["TANZANIA_SNOWFLAKE_DATABASE"] = snowflake_secret["tanzania_database"]
    os.environ["KENYA_SNOWFLAKE_DATABASE"] = snowflake_secret["kenya_database"]
    os.environ["SNOWFLAKE_INGESTION_USERNAME"] = snowflake_users_secret['AROUSI_USERNAME']
    os.environ["SNOWFLAKE_INGESTION_PASSWORD"] = snowflake_users_secret["OMAR_ALAROUSI"]
    
    # ########## DBT SNOWFLAKE credentials ##########
    os.environ["SNOWFLAKE_DBT_USERNAME"] = snowflake_dbt_maxab["username"]
    os.environ["SNOWFLAKE_DBT_PASSWORD"] = snowflake_dbt_maxab["password"]

    ########### MAXROUTE credentials ##########
    os.environ["MAXROUTE_DWH_READER_HOST"] = maxroute_dwh_secret["host_reader"]
    os.environ["MAXROUTE_DWH_READER_NAME"] = maxroute_dwh_secret["dbname"]
    os.environ["MAXROUTE_DWH_READER_USER_NAME"] = maxroute_dwh_secret["username"]
    os.environ["MAXROUTE_DWH_READER_PASSWORD"] = maxroute_dwh_secret["password"]
    os.environ["MAXROUTE_DWH_WRITER_HOST"] = maxroute_dwh_secret["host_writer"]
    os.environ["MAXROUTE_DWH_WRITER_NAME"] = maxroute_dwh_secret["dbname"]
    os.environ["MAXROUTE_DWH_WRITER_USER_NAME"] = maxroute_dwh_secret["username"]
    os.environ["MAXROUTE_DWH_WRITER_PASSWORD"] = maxroute_dwh_secret["password"]
    
    os.environ["morocco_metabase_user"] = morocco_metabase_access["username"]
    os.environ["morocco_metabase_password"] = morocco_metabase_access["password"]
        
    # ########## THREE_CX credentials ##########
    os.environ['THREE_CX_HOST'] = three_cx_egypt_secret['host']
    os.environ['THREE_CX_HOST_PORT'] = three_cx_egypt_secret['host_port']
    os.environ['THREE_CX_USERNAME'] = three_cx_egypt_secret['username']
    os.environ['THREE_CX_PASSWORD'] = three_cx_egypt_secret['password']
    os.environ['THREE_CX_DB_NAME'] = three_cx_egypt_secret['dbname']
    os.environ['THREE_CX_REMOTE_PORT'] = three_cx_egypt_secret['remote_port']
    os.environ['THREE_CX_LOCAL_PORT'] = three_cx_egypt_secret['local_port']
    
    ########### Mongo credentials ##########
    os.environ["MONGO_CONNECTION"] = mongo_creds["mongodb_connection"]

    ########### ODOO SHAMSELKHEIR Portal credentials ##########
    os.environ["SHAMSELKHEIR_ODOO_URL"] = shamselkheir_odoo_secret["url"]
    os.environ["SHAMSELKHEIR_ODOO_DB"] = shamselkheir_odoo_secret["db"]
    os.environ["SHAMSELKHEIR_ODOO_USERNAME"] = shamselkheir_odoo_secret["username"]
    os.environ["SHAMSELKHEIR_ODOO_PASSWORD"] = shamselkheir_odoo_secret["password"]
    
    ########### MAXSUPPORT credentials ##########
    os.environ["MAXSUPPORT_HOST"] = maxsupport_secret["host"]
    os.environ["MAXSUPPORT_NAME"] = maxsupport_secret["dbname"]
    os.environ["MAXSUPPORT_USERNAME"] = maxsupport_secret["username"]
    os.environ["MAXSUPPORT_PASSWORD"] = maxsupport_secret["password"]
    
    ########### SNOWFLAKE INGESTION credentials ##########
    snowflake_ingestion_secrets = json.loads(get_secret("prod/snowflake/ingestion"))

    os.environ["ingestion_key_bucket"] = snowflake_ingestion_secrets["private_key_bucket"]
    os.environ["ingestion_key_path"] = snowflake_ingestion_secrets["private_key_path"]
    os.environ["ingestion_pass"] = snowflake_ingestion_secrets["encryption_password"]
    os.environ["ingestion_user"] = snowflake_ingestion_secrets["username"]
    os.environ["ingestion_account"] = snowflake_ingestion_secrets["account"]
    os.environ["ingestion_role"] = snowflake_ingestion_secrets["role"]
    
    ########### FINTECH_EMONEY credentials ##########
    os.environ["FINTECH_EMONEY_EMAIL"] = fintech_service_account["email_name"]
    os.environ["FINTECH_EMONEY_PASSWORD"] = fintech_service_account["email_password"]

    s3 = boto3.client('s3')

    bucket_name = os.environ["ingestion_key_bucket"]
    key_name = os.environ["ingestion_key_path"]
    s3.download_file(bucket_name, key_name, '/tmp/rsa_key.p8')
    
    # ########## BigQuery credentials ##########
    json_path = str(Path.home())+"/service_account_key.json"
    print(json_path)
    bigquery_key = get_secret("prod/bigquery/sagemaker")
    f = open(json_path, "w")
    f.write(bigquery_key)
    f.close()

    # ########## GOOGLE SHEETS credentials ##########
    json_path_sheets = str(Path.home())+"/service_account_key_sheets.json"
    sheets_key = get_secret("prod/maxab-sheets")
    f = open(json_path_sheets, "w")
    f.write(sheets_key)
    f.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS_SHEETS"] = json_path_sheets

    # ########## SLACK credentials ##########
    slack_secret = json.loads(get_secret("prod/slack/reports"))
    os.environ["SLACK_TOKEN"] = slack_secret["token"]


def google_sheets(workbook, sheet, action,cols=[], df=None):
    from oauth2client.service_account import ServiceAccountCredentials
    import gspread
    from gspread_dataframe import get_as_dataframe, set_with_dataframe
    import os
    import pandas as pd

    initialize_env()

    scope = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive.file',
        'https://www.googleapis.com/auth/drive'
    ]
    
    creds = ServiceAccountCredentials.from_json_keyfile_name(os.environ["GOOGLE_APPLICATION_CREDENTIALS_SHEETS"] , scope)
    client = gspread.authorize(creds)
    wks = client.open(workbook).worksheet(sheet)
    
    if action.lower() == 'get':
        if len(cols) > 0:
            sheet = get_as_dataframe(
                wks, 
                parse_dates=True,
                usecols = cols,
                evaluate_formulas=True
            ).dropna(how = 'all')
        elif len(cols) == 0:
            sheet = get_as_dataframe(
                wks,
                parse_dates=True,
                evaluate_formulas=True
            ).dropna(how = 'all')
            
        return sheet
    elif action.lower() == 'overwrite':
        wks.clear()
        set_with_dataframe(wks, df)
        return('data is added to the sheet successfully')
    elif action.lower() == 'append':
        existing_data = get_as_dataframe(wks)
        updated_data = pd.concat([existing_data, df], ignore_index=True)
        set_with_dataframe(wks, updated_data)
        return 'Data is appended to the sheet successfully'
    elif action.lower() == 'clear':
        wks.clear()
        return('sheet is cleared')


def send_text_slack(channel, text):
    import slack
    import os

    initialize_env()

    client = slack.WebClient(token=os.environ["SLACK_TOKEN"])
    try:
        client.chat_postMessage(
        channel=channel,
        text=text
      )
        print('Message Sent')
    except Exception as e:
        raise e


def task_fail_slack_alert(context):
    slack_msg = """
        :red_circle: Task Failed.
        *Task*: {task}  
        *Dag*: {dag} 
        *Execution Time*: {exec_date}  
        *Reason*: {exception}
    """.format(
        task=context.get('task_instance').task_id,
        dag=context.get('task_instance').dag_id,
        exec_date=context.get('execution_date'),
        exception=context.get('exception')
    )

    send_text_slack(channel='airflow_alerts', text=slack_msg)


def dwh_query(country, query, action, columns=[], conn=None):
    import psycopg2
    import pandas as pd
    import os

    initialize_env()
    
    try:
        if action.lower() == 'read':
            credentials_action = 'READER'
        elif action.lower() == 'write':
            credentials_action = 'WRITER'
        else:
            logger.error("Invalid action")
            raise ValueError("Invalid action")
        
        if conn:
            host = conn['host']
            database = conn['database']
            user = conn['user']
            password = conn['password']
        
        else:
            host = os.environ[f'{country.upper()}_DWH_{credentials_action}_HOST']
            database = os.environ[f'{country.upper()}_DWH_{credentials_action}_NAME']
            user = os.environ[f'{country.upper()}_DWH_{credentials_action}_USER_NAME']
            password = os.environ[f'{country.upper()}_DWH_{credentials_action}_PASSWORD']

        # Establish database connection
        with psycopg2.connect(host=host, database=database, user=user, password=password) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                if action == 'read':
                    data = cur.fetchall()
                    column_names = [desc[0] for desc in cur.description]
                    if len(columns) == 0:
                        out = pd.DataFrame(data, columns=column_names)
                    else:
                        out = pd.DataFrame(data, columns=columns)
                    logger.info("Data retrieved successfully", exc_info=True)
                    return out
                elif action == 'write':
                    conn.commit()
                    logger.info("Query executed successfully", exc_info=True)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise


def upload_dataframe_to_pg(df, country, if_exists, schema_name, table_name, db_conn=None): 
    from sqlalchemy import create_engine
    import pandas as pd
    import os
    
    initialize_env()
    if db_conn:
        host = db_conn['host']
        database = db_conn['database']
        user = db_conn['user']
        password = db_conn['password']
    else:
        host = os.environ[f'{country.upper()}_DWH_WRITER_HOST']
        database = os.environ[f'{country.upper()}_DWH_WRITER_NAME']
        user = os.environ[f'{country.upper()}_DWH_WRITER_USER_NAME']
        password = os.environ[f'{country.upper()}_DWH_WRITER_PASSWORD']

    engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}/{database}')
        
    df.to_sql(table_name.lower(), schema=schema_name.lower(), con=engine, if_exists=if_exists.lower(), index=False)


def snowflake_query(country, query, warehouse=None, columns=[], conn=None):
    import snowflake.connector
    import os
    import pandas as pd
    import numpy as np

    initialize_env()
    
    if conn:
        con = conn
    else:
        config = {
            'user': os.environ["ingestion_user"],
            'account': os.environ["ingestion_account"],
            'private_key_file': '/tmp/rsa_key.p8',
            'private_key_file_pwd': os.environ["ingestion_pass"].encode('utf-8'),
            'database': os.environ[f"{country.upper()}_SNOWFLAKE_DATABASE"],
            'role': os.environ["ingestion_role"],
            'schema': 'PUBLIC'
        }
        
        con = snowflake.connector.connect(**config)

    try:
        cur = con.cursor()
        if warehouse:
            cur.execute(f'USE WAREHOUSE {os.environ[f"{warehouse}"]}')
        else:
            cur.execute(f'USE WAREHOUSE {os.environ["SNOWFLAKE_AIRFLOW_WAREHOUSE"]}')
        
        cur.execute(query)
        
        column_names = [col[0] for col in cur.description]
        
        results = cur.fetchall()
        
        if not results:
            out = pd.DataFrame(columns=[name.lower() for name in column_names])
        else:
            if len(columns) == 0:
                out = pd.DataFrame(np.array(results), columns=column_names)
                out.columns = out.columns.str.lower()
            else:
                out = pd.DataFrame(np.array(results), columns=columns)
                out.columns = out.columns.str.lower()
        
        return out
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise
    finally:
        cur.close()
        con.close()


def upload_dataframe_to_snowflake(country, df, schema_name, table_name, method, auto_create_table=True, conn=None):
    import snowflake.connector
    from snowflake.connector.pandas_tools import write_pandas
    import os

    initialize_env()
    
    schema_name = schema_name.upper()
    table_name = table_name.upper()
    df.columns = df.columns.str.upper()
    
    if conn:
        conn = conn
    else:
        conn = snowflake.connector.connect(
            user=os.environ["ingestion_user"],
            account=os.environ["ingestion_account"],
            private_key_file='/tmp/rsa_key.p8',
            private_key_file_pwd=os.environ["ingestion_pass"].encode('utf-8'),
            database=os.environ[f"{country.upper()}_SNOWFLAKE_DATABASE"],
            role=os.environ["ingestion_role"],
            warehouse=os.environ["SNOWFLAKE_AIRFLOW_WAREHOUSE"]
        )
    
    success = 'Failed'

    if method.lower() == 'append':
        # this will append new rows to existing rows 
        success, _, _, _ = write_pandas(
            conn=conn,
            df=df,
            table_name=table_name,
            schema=schema_name,
            use_logical_type=True,
            overwrite=False,
            auto_create_table=auto_create_table
        )
    
    elif method.lower() == 'replace':
        # this will remove all the records and add new records without dropping the existing table
        # it is used when only need to replace the existing data with new data 
        cur = conn.cursor()
        cur.execute(f'truncate table {schema_name}.{table_name};')
        cur.close()
        success, _, _, _ = write_pandas(
            conn=conn,
            df=df,
            table_name=table_name,
            schema=schema_name,
            use_logical_type=True,
            overwrite=False,
            auto_create_table=auto_create_table
        )
    elif method.lower() == 'overwrite':
        # this will drop the table and create a new table with the new data
        # It is used when need to change the structure of the table 
        success, _, _, _ = write_pandas(
            conn=conn,
            df=df,
            table_name=table_name,
            schema=schema_name,
            use_logical_type=True,
            overwrite=True,
            auto_create_table=auto_create_table
        )

    conn.close()
    return success


def ret_metabase(country, question, filters={}):
    from io import StringIO
    import requests
    import pandas as pd
    import json
    import os

    initialize_env()
    
    question_id = str(question)
    
    if country.lower() == 'egypt':
        base_url = 'https://bi.maxab.info/api'
        username = str(os.environ["EGYPT_METABASE_USERNAME"])
        password = str(os.environ["EGYPT_METABASE_PASSWORD"])
    else:
        base_url = 'https://bi.maxabma.com/api'
        username = str(os.environ["AFRICA_METABASE_USERNAME"])
        password = str(os.environ["AFRICA_METABASE_PASSWORD"])

    base_headers = {'Content-Type': 'application/json'}

    try:
        s_response = requests.post(
            base_url + '/session',
            data=json.dumps({
                'username': username,
                'password': password
            }),
            headers=base_headers)
        
        s_response.raise_for_status()

        session_token = s_response.json()['id']
        base_headers['X-Metabase-Session'] = session_token
        
        params = []
        
        for name, value in filters.items():
            filter_type, filter_value = value
            param = {'target': ['variable', ['template-tag', name]], 'value': filter_value}
            
            if filter_type.lower() == 'date':
                param['type'] = 'date/range' if isinstance(filter_value, list) else 'date/single'
            elif filter_type.lower() == 'category':
                param['type'] = 'category'
            elif filter_type.lower() == 'text':
                param['type'] = 'text'
            elif filter_type.lower() == 'number':
                param['type'] = 'number'
            elif filter_type.lower() == 'field list':
                param['type'] = 'id'
                param['target'] = ['dimension', ['template-tag', name]]
            
            params.append(param)

        p_response = requests.post(base_url + '/card/' + question_id + '/query/csv', 
                                   json={'parameters': params}, 
                                   headers=base_headers)
        p_response.raise_for_status()

        my_dict = p_response.content
        s = str(my_dict, 'utf-8')
        my_dict = StringIO(s)
        df = pd.read_csv(my_dict)
        return(df)
    
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise

def processing_jobs(input_s3, output_s3, job_name,role=None, image=None, instance_type=None):
    
    from datetime import datetime 
    import nbformat

    # Define your AWS and SageMaker configurations
    if role:
        ROLE_ARN = role
    else:
        ROLE_ARN = "arn:aws:iam::876425898567:role/service-role/AmazonSageMaker-ExecutionRole-20220614T112152"

    if image:
        IMAGE_URI=image
    else:
        IMAGE_URI = f"876425898567.dkr.ecr.us-east-1.amazonaws.com/processing-jobs:latest"
    
    if instance_type:
        InstanceType=instance_type
    else:
        InstanceType="ml.t3.medium"
        
    INPUT_S3_URI = input_s3
    OUTPUT_S3_URI = output_s3
    notebook = input_s3.split('/')[-1].split('.')[0]
    
    # Define SageMaker Processing job configuration
    PROCESSING_JOB_CONFIG = {
        "ProcessingJobName": job_name,
        "AppSpecification": {
            "ImageUri": IMAGE_URI,
            "ContainerEntrypoint": [
                "papermill", 
                f"/opt/ml/processing/input/{notebook}.ipynb", 
                f"/opt/ml/processing/output/out-{job_name}.ipynb",
                "--kernel", "python3"],
        },
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": InstanceType,
                "VolumeSizeInGB": 30,
            }
        },
        "NetworkConfig":{
        'VpcConfig': {
            'SecurityGroupIds': ['sg-0f79a2829997bc207'],
            'Subnets': ['subnet-0e4ad491dc9bc5d6a']
            }
        },
        "RoleArn": ROLE_ARN,
        "ProcessingInputs": [
            {
                "InputName": "input-1",
                "S3Input": {
                    "S3Uri": INPUT_S3_URI,
                    "LocalPath": "/opt/ml/processing/input",  # Local path inside the container
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            }
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "output-1",
                    "S3Output": {
                        "S3Uri": OUTPUT_S3_URI,
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    }
                }
            ]
        },
    }
    
    return PROCESSING_JOB_CONFIG