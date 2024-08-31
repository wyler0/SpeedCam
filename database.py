# © 2024 Wyler Zahm. All rights reserved.

import os
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session
from contextlib import contextmanager
from models import Base  # Import the Base from your models file

class SessionFactory:
    engine: Engine = None
    SessionLocal: sessionmaker = None
    
    def __init__(self, db_url: str):
        connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
        self.engine = create_engine(db_url, connect_args=connect_args)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def create_database(self):
        Base.metadata.create_all(bind=self.engine)

    @contextmanager
    def get_db_with(self) -> Session:
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

# Lazy initialization of the default session factory
_default_session_factory = None

def set_database_url(db_url: str):
    global _default_session_factory  # Add this line to modify the global variable
    os.environ["DATABASE_URL"] = db_url
    _default_session_factory = None
    get_default_session_factory()
    
def get_database_url() -> str:
    if os.getenv("DATABASE_URL", "sqlite:///./vehicle_tracking.db") is None:
        os.environ["DATABASE_URL"] = "sqlite:///./vehicle_tracking.db"
    return os.getenv("DATABASE_URL", "sqlite:///./vehicle_tracking.db")

def get_default_session_factory() -> SessionFactory:
    global _default_session_factory
    if _default_session_factory is None:
        db_url = get_database_url()
        _default_session_factory = SessionFactory(db_url)
        _default_session_factory.create_database()
    return _default_session_factory

# Dependency for FastAPI (if you're using it)
def get_db():
    session_factory = get_default_session_factory()
    with session_factory.get_db_with() as db:
        yield db