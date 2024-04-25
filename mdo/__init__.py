from .movie_director_office import (
    MDO_Researcher, MDO_Statistician, MDO_Agent, MDO_Forecaster,
    LinearRegression, make_pipeline, StandardScaler, MLPClassifier,
    RandomForestClassifier, RandomForestRegressor,
    pd
)

print(f'Welcome to the Movie Director Office! 🎬')
print(f'Please, meet our team: Researcher, Statistician, Agent, and Forecaster.')

__all__ = [
    'MDO_Researcher', 'MDO_Statistician', 'MDO_Agent', 'MDO_Forecaster',
    'LinearRegression', 'make_pipeline', 'StandardScaler', 'MLPClassifier',
    'RandomForestClassifier', 'RandomForestRegressor',
    'pd'
]
