import pytest


# This fixture is accessible to all test cases
@pytest.fixture
def sample_fixture():
    return "sample fixture"


@pytest.fixture
def weight_data():
    pass
