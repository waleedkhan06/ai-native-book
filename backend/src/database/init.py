from .models import Base
from .session import engine
import logging

logger = logging.getLogger(__name__)


def init_db():
    """
    Initialize the database by creating all tables
    """
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


if __name__ == "__main__":
    init_db()