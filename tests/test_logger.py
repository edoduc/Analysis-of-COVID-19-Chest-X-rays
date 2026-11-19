import logging
import sys

import pytest

from src.logger import get_logger


@pytest.fixture(autouse=True)
def nettoyer_les_loggers():
    """Après chaque test, on retire les handlers des loggers créés ici."""
    yield
    for nom in list(logging.Logger.manager.loggerDict):
        if nom.startswith("test_logger_"):
            logging.getLogger(nom).handlers.clear()


def test_get_logger_renvoie_un_logger_avec_le_bon_nom():
    logger = get_logger("test_logger_nom")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger_nom"


def test_get_logger_nom_par_defaut():
    assert get_logger().name == "covid_xray"


def test_get_logger_configuration_par_defaut():
    logger = get_logger("test_logger_defaut")
    assert logger.level == logging.INFO
    assert logger.propagate is False
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_get_logger_appele_deux_fois_ne_duplique_pas_les_handlers():
    premier = get_logger("test_logger_double")
    second = get_logger("test_logger_double")
    assert premier is second
    assert len(second.handlers) == 1


def test_get_logger_niveau_personnalise():
    logger = get_logger("test_logger_niveau", level=logging.DEBUG)
    assert logger.level == logging.DEBUG


def test_get_logger_format_du_message(capsys):
    logger = get_logger("test_logger_format")
    logger.info("bonjour")
    sortie = capsys.readouterr().out
    assert "[INFO]" in sortie
    assert "test_logger_format" in sortie
    assert "bonjour" in sortie


def test_get_logger_respecte_le_niveau(capsys):
    logger = get_logger("test_logger_filtre", level=logging.WARNING)
    logger.info("message ignore")
    logger.warning("message affiche")
    sortie = capsys.readouterr().out
    assert "message ignore" not in sortie
    assert "message affiche" in sortie