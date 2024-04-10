"Tests for the plot_flights_from_country method of the FeatherFlight class."
# Built-in imports
from unittest.mock import patch
import os
import sys

# Third-party imports
import pandas as pd
import pytest

# Local application/library specific imports
sys.path.append('../scripts/')
# Adjust this import to your actual module path
from featherflight import FeatherFlight

# Prepare mock data with correct data types for airports and routes
AIRPORTS_DATA = {
    "Airport ID": ["1", "2", "3", "4"],
    "Country": ["CountryA", "CountryA", "CountryB", "CountryB"],
    "Latitude": [10.0, 20.0, 30.0, 40.0],
    "Longitude": [50.0, 60.0, 70.0, 80.0],
}
AIRPORTS_DF = pd.DataFrame(AIRPORTS_DATA)

ROUTES_DATA = {
    "Source airport ID": ["1", "2", "3", "1"],
    "Destination airport ID": ["2", "3", "4", "3"],
    "Stops": [0, 0, 0, 0],
}
ROUTES_DF = pd.DataFrame(ROUTES_DATA)


@pytest.fixture
def feather_flight_instance():
    """Fixture to create a FeatherFlight instance with mock data."""
    ff = FeatherFlight()
    ff.airports = AIRPORTS_DF
    ff.routes = ROUTES_DF
    ff.calculate_distances_routes()
    return ff

# After making the plot interactive with folium, the matplotlib show method is not called anymore

# @patch('matplotlib.pyplot.show')
# def test_internal_flights_only(mock_show, feather_flight_instance):
#     """Ensure only internal flights are included when internal=True."""
#     feather_flight_instance.plot_flights_from_country(
#         "CountryA", internal=True)
#     mock_show.assert_called_once()


# @patch('matplotlib.pyplot.show')
# def test_external_flights_included(mock_show, feather_flight_instance):
#     """Verify that flights to external destinations are included when internal=False."""
#     feather_flight_instance.plot_flights_from_country(
#         "CountryA", internal=False)
#     mock_show.assert_called_once()


def test_country_not_in_dataset_raises_error(feather_flight_instance):
    """Check that an error is raised for a non-existing country."""
    with pytest.raises(AssertionError):
        feather_flight_instance.plot_flights_from_country(
            "CountryZ", internal=True)


# @patch('matplotlib.pyplot.show')
# def test_plot_flights_from_non_existing_country(mock_show, feather_flight_instance):
#     """Ensure an error is raised when attempting to plot flights from a non-existing country."""
#     with pytest.raises(AssertionError):
#         feather_flight_instance.plot_flights_from_country(
#             "NonExistingCountry", internal=True)
#     mock_show.assert_not_called()


# @patch('matplotlib.pyplot.show')
# def test_plot_flights_from_existing_country(mock_show, feather_flight_instance):
#     """Verify plotting does not raise errors for an existing country."""
#     feather_flight_instance.plot_flights_from_country("CountryA")
#     mock_show.assert_called_once()
