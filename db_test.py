import pandas as pd
import numpy as np
from sqlalchemy import create_engine

dbname = 'frontpage'
username = 'ubuntu'
engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
sql_query = "SELECT * FROM frontpage;" # WHERE article_order <= 10;"
frontpage_data = pd.read_sql_query(sql_query,engine)

print frontpage_data
