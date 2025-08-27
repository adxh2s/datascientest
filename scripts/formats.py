import json
from bs4 import BeautifulSoup
from typing import Optional

def format_json(url: str, html: str) -> str:
    """Formate le contenu dans un objet JSON."""
    soup = BeautifulSoup(html, 'html.parser')
    return json.dumps({'url': url, 'contenu': soup.prettify()}, ensure_ascii=False)

def format_texte(html: str) -> str:
    """Extrait le texte brut du HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()

def format_html(html: str) -> str:
    """Retourne le contenu HTML brut."""
    return html

def format_db(url: str, html: str, db_type: str, host: str, port: str, db_name: str, db_user: str, db_password: str) -> str:
    """Enregistre le contenu dans une base de données selon le type spécifié."""
    db_type = db_type.lower()
    if db_type == 'postgresql':
        return format_db_postgresql(url, html, host, port, db_name, db_user, db_password)
    elif db_type == 'mysql':
        return format_db_mysql(url, html, host, port, db_name, db_user, db_password)
    elif db_type == 'oracle':
        return format_db_oracle(url, html, host, port, db_name, db_user, db_password)
    else:
        raise ValueError(f"Type de base de données non supporté : {db_type}")

def format_db_postgresql(url: str, html: str, host: str, port: str, db_name: str, db_user: str, db_password: str) -> str:
    """Enregistre le contenu dans une base de données PostgreSQL."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=db_name,
            user=db_user,
            password=db_password
        )
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS web_content (
                id SERIAL PRIMARY KEY,
                url TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        soup = BeautifulSoup(html, 'html.parser')
        cursor.execute(
            "INSERT INTO web_content (url, content) VALUES (%s, %s)",
            (url, soup.prettify())
        )
        conn.commit()
        conn.close()
        return "Contenu enregistré dans la base de données PostgreSQL."
    except Exception as e:
        raise Exception(f"Erreur lors de l'insertion en base PostgreSQL : {str(e)}")

def format_db_mysql(url: str, html: str, host: str, port: str, db_name: str, db_user: str, db_password: str) -> str:
    """Enregistre le contenu dans une base de données MySQL."""
    try:
        import mysql.connector
        conn = mysql.connector.connect(
            host=host,
            port=port,
            database=db_name,
            user=db_user,
            password=db_password
        )
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS web_content (
                id INT AUTO_INCREMENT PRIMARY KEY,
                url TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        soup = BeautifulSoup(html, 'html.parser')
        cursor.execute(
            "INSERT INTO web_content (url, content) VALUES (%s, %s)",
            (url, soup.prettify())
        )
        conn.commit()
        conn.close()
        return "Contenu enregistré dans la base de données MySQL."
    except Exception as e:
        raise Exception(f"Erreur lors de l'insertion en base MySQL : {str(e)}")

def format_db_oracle(url: str, html: str, host: str, port: str, db_name: str, db_user: str, db_password: str) -> str:
    """Enregistre le contenu dans une base de données Oracle."""
    try:
        import cx_Oracle
        dsn = cx_Oracle.makedsn(host, port, service_name=db_name)
        conn = cx_Oracle.connect(user=db_user, password=db_password, dsn=dsn)
        cursor = conn.cursor()
        cursor.execute("""
            BEGIN
                EXECUTE IMMEDIATE 'CREATE TABLE web_content (
                    id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    url CLOB,
                    content CLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )';
            EXCEPTION
                WHEN OTHERS THEN
                    IF SQLCODE != -955 THEN
                        RAISE;
                    END IF;
            END;
        """)
        soup = BeautifulSoup(html, 'html.parser')
        cursor.execute(
            "INSERT INTO web_content (url, content) VALUES (:1, :2)",
            (url, soup.prettify())
        )
        conn.commit()
        conn.close()
        return "Contenu enregistré dans la base de données Oracle."
    except Exception as e:
        raise Exception(f"Erreur lors de l'insertion en base Oracle : {str(e)}")
