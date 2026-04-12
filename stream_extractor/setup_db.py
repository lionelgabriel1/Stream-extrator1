"""
Setup interativo da base de dados PostgreSQL.
Cria o ficheiro .env e inicializa as tabelas.
Execute uma vez antes de iniciar o sistema:
    python setup_db.py
"""

import os
import sys
import getpass
from pathlib import Path

ENV_PATH = Path(__file__).parent / ".env"


def ask(prompt: str, default: str = "", secret: bool = False) -> str:
    if default:
        display = f"{prompt} [{default}]: "
    else:
        display = f"{prompt}: "

    if secret:
        value = getpass.getpass(display)
    else:
        value = input(display).strip()

    return value if value else default


def write_env(config: dict):
    lines = [
        "# Gerado automaticamente pelo setup_db.py",
        "# NÃO adicionar ao git\n",
        "# ─── Base de Dados PostgreSQL ───────────────────────────────",
        f"DB_HOST={config['host']}",
        f"DB_PORT={config['port']}",
        f"DB_NAME={config['name']}",
        f"DB_USER={config['user']}",
        f"DB_PASSWORD={config['password']}",
        "",
        "# ─── Stream ─────────────────────────────────────────────────",
        f"STREAM_URL={config['stream_url']}",
        f"STREAM_QUALITY={config['stream_quality']}",
    ]
    ENV_PATH.write_text("\n".join(lines))
    print(f"\n  ✓ Ficheiro .env criado em: {ENV_PATH}")


def create_database(config: dict) -> bool:
    """Tenta criar a base de dados se não existir."""
    try:
        import psycopg2
        from psycopg2 import sql
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

        # Liga ao postgres (base de dados padrão) para criar a nova
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            dbname='postgres',
            user=config['user'],
            password=config['password'],
            connect_timeout=10
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Verifica se já existe
        cursor.execute("SELECT 1 FROM pg_database WHERE datname=%s", (config['name'],))
        exists = cursor.fetchone()

        if exists:
            print(f"  ✓ Base de dados '{config['name']}' já existe")
        else:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(
                sql.Identifier(config['name'])
            ))
            print(f"  ✓ Base de dados '{config['name']}' criada")

        cursor.close()
        conn.close()
        return True

    except ImportError:
        print("  ⚠ psycopg2 não instalado. Instala com: pip install psycopg2-binary")
        return False
    except Exception as e:
        print(f"  ✗ Erro ao criar base de dados: {e}")
        return False


def create_tables(config: dict) -> bool:
    """Cria as tabelas na base de dados."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))

        # Injeta config temporariamente
        os.environ['DB_HOST']     = config['host']
        os.environ['DB_PORT']     = config['port']
        os.environ['DB_NAME']     = config['name']
        os.environ['DB_USER']     = config['user']
        os.environ['DB_PASSWORD'] = config['password']

        from database.manager import DatabaseManager
        db_config = {
            "type":     "postgresql",
            "host":     config['host'],
            "port":     int(config['port']),
            "name":     config['name'],
            "user":     config['user'],
            "password": config['password'],
        }
        DatabaseManager(db_config)
        print("  ✓ Tabelas criadas com sucesso")
        print("    → matches")
        print("    → extraction_log")
        print("    → stream_status")
        return True

    except Exception as e:
        print(f"  ✗ Erro ao criar tabelas: {e}")
        return False


def main():
    print("""
╔══════════════════════════════════════════════════════╗
║     STREAM STATS EXTRACTOR — Setup PostgreSQL        ║
╚══════════════════════════════════════════════════════╝
""")

    # Carrega .env existente como defaults
    defaults = {
        'host': 'localhost', 'port': '5432',
        'name': 'stream_stats', 'user': 'postgres',
        'password': '',
        'stream_url': 'https://www.twitch.tv/esbfootball',
        'stream_quality': '720p',
    }

    if ENV_PATH.exists():
        print("  Ficheiro .env encontrado. A usar valores existentes como padrão.\n")
        for line in ENV_PATH.read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                key, _, val = line.partition('=')
                mapping = {
                    'DB_HOST': 'host', 'DB_PORT': 'port',
                    'DB_NAME': 'name', 'DB_USER': 'user',
                    'DB_PASSWORD': 'password',
                    'STREAM_URL': 'stream_url',
                    'STREAM_QUALITY': 'stream_quality',
                }
                if key in mapping:
                    defaults[mapping[key]] = val

    print("─── Configuração PostgreSQL ──────────────────────────")
    config = {
        'host':           ask("Host",              defaults['host']),
        'port':           ask("Porto",             defaults['port']),
        'name':           ask("Nome da base de dados", defaults['name']),
        'user':           ask("Utilizador",        defaults['user']),
        'password':       ask("Password",          defaults['password'], secret=True),
        'stream_url':     ask("\nURL da stream",   defaults['stream_url']),
        'stream_quality': ask("Qualidade",         defaults['stream_quality']),
    }

    print("\n─── A configurar ─────────────────────────────────────")

    # 1. Escreve .env
    write_env(config)

    # 2. Cria base de dados
    print("\n  A criar base de dados...")
    db_ok = create_database(config)

    # 3. Cria tabelas
    if db_ok:
        print("\n  A criar tabelas...")
        create_tables(config)

    print("""
─── Concluído ────────────────────────────────────────

  Para iniciar o sistema:
    python main.py

  Para testar OCR num screenshot:
    python test_ocr.py screenshot.jpg

══════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    main()
