import psycopg2
from decouple import config

conn = psycopg2.connect(
    host=config('DB_HOST'),
    database=config('DB_NAME'),
    user=config('DB_USER'),
    password=config('DB_PASSWORD'),
    port=config('DB_PORT')
)

def get_connection():
    return conn

# if __name__ == '__main__':
#     cur = conn.cursor()

#     cur.execute("SELECT version();")
#     db_version = cur.fetchone()
#     print(f"Connected to database version: {db_version}")


#     # Close the cursor and connection
#     cur.close()
#     conn.close()